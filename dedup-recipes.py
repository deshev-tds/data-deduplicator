import sys
import subprocess
import importlib
import os
import json
import time
import random
import csv
import re
import difflib # <--- NEW: For typo-similarity checks (SequenceMatcher ratio; not Levenshtein distance as it was in the previous version)

# =========================
# FIX FOR MACOS M-SERIES
# =========================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ========================
# 1. AUTO-INSTALL PREREQUISITES
# =========================
REQUIRED_PACKAGES = [
    "torch", 
    "sentence-transformers", 
    "pandas", 
    "networkx", 
    "tqdm",
    "scikit-learn"
]

def install_packages():
    needs_install = []
    for package in REQUIRED_PACKAGES:
        pkg_import_name = package.replace("-", "_").split("=")[0]
        if pkg_import_name == "scikit_learn": pkg_import_name = "sklearn"
        try:
            importlib.import_module(pkg_import_name)
        except ImportError:
            needs_install.append(package)
    
    if needs_install:
        print(f"Installing missing packages: {', '.join(needs_install)}...")
        for pkg in needs_install:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        print("Dependencies installed.\n")

install_packages()

# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import torch
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
INPUT_FILE = "bg-recipes-final.tsv"   
OUTPUT_FILE = "bg-recipes-deduplicated_final.tsv"

# REPORTING FILES
REPORT_MERGES = "report_merges.csv"            
REPORT_NEAR_MISSES = "report_near_misses.csv"  
REPORT_SCATTER = "report_scatter_plot.csv"     
REPORT_GRAPH_EDGES = "report_graph_edges.csv"  

# TUNING PARAMETERS
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SIMILARITY_THRESHOLD = 0.94       
INGREDIENT_OVERLAP_THRESHOLD = 0.50 

# VETO THRESHOLD (TF-IDF title similarity)
# Raised from 0.40 to 0.55 after introducing stop-word stripping in titles.
# Example: boilerplate-heavy titles with different key nouns tend to drop below this.
TITLE_SAFETY_THRESHOLD = 0.55  

BATCH_SIZE = 64
SEARCH_K = 15

# DISPLAY FORMAT
BAR_FMT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

# NEW: Stop words for Titles only (Noise reduction)
STOP_WORDS_TITLE = {'и', 'в', 'с', 'на', 'за', 'по', 'от', 'до', 'или', 'а', 'но', 'че', 'към', 'върху', 'без'}

# =========================
# CORE LOGIC
# =========================

def setup_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def normalize_title(text):
    if not isinstance(text, str): return ""
    clean = "".join(c for c in text.lower() if c.isalnum() or c.isspace())
    return " ".join(clean.split())

def get_anchor_text(text):
    """
    Cleans title text for TF-IDF title-similarity veto.
    Removes high-frequency 'noise' words (prepositions/conjunctions) to expose defining tokens.
    """
    norm = normalize_title(text)
    tokens = norm.split()
    clean_tokens = [t for t in tokens if t not in STOP_WORDS_TITLE]
    if not clean_tokens: # Fallback if title is just "Salad with..."
        return norm
    return " ".join(clean_tokens)

def parse_ingredients(products_str):
    if not isinstance(products_str, str) or pd.isna(products_str):
        return []
    return [x.strip().lower() for x in products_str.split(',') if x.strip()]

STOP_INGREDIENTS = {
    'сол', 'захар', 'черен пипер', 'олио', 'вода', 'брашно', 
    'яйца', 'лук', 'чесън', 'магданоз', 'червен пипер', 
    'кисело мляко', 'сода', 'бакпулвер', 'оцет', 'масло'
}

def calculate_jaccard(list_a, list_b):
    set_a = set(x.lower().strip() for x in list_a)
    set_b = set(x.lower().strip() for x in list_b)
    
    set_a_clean = set_a - STOP_INGREDIENTS
    set_b_clean = set_b - STOP_INGREDIENTS
    
    if len(set_a_clean) == 0 or len(set_b_clean) == 0:
        set_a_clean = set_a
        set_b_clean = set_b

    if not set_a_clean or not set_b_clean: return 0.0
    
    intersection = len(set_a_clean.intersection(set_b_clean))
    union = len(set_a_clean.union(set_b_clean))
    
    if union == 0: return 0.0
    return intersection / union

def resolve_cluster(records, reason="Unknown"):
    def get_len(rec, field):
        val = rec.get(field, "")
        return len(str(val)) if not pd.isna(val) else 0

    survivor = sorted(
        records, 
        key=lambda x: (
            x.get('quality_flag', 'unknown') == 'ok', 
            get_len(x, 'description'),                
            len(parse_ingredients(x.get('products_extracted', ''))) 
        ), 
        reverse=True
    )[0]
    return survivor

def main():
    print("\n" + "="*60)
    print("HYBRID DEDUPLICATION ENGINE v4.0 (Stop-Word Veto)")
    print("="*60 + "\n")
    
    device = setup_device()
    print(f"-> Hardware Acceleration: {device.upper()}")

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file {INPUT_FILE} not found.")
        return

    print(f"\n[Phase 1] Ingesting TSV Dataset...")
    try:
        df = pd.read_csv(INPUT_FILE, sep='\t')
    except Exception as e:
        print(f"Error reading TSV: {e}")
        return

    df.fillna("", inplace=True)
    if 'id' not in df.columns: df['id'] = range(len(df))
    
    initial_count = len(df)
    print(f"-> Loaded {initial_count:,} records.")
    
    merge_log, scatter_log, near_miss_log, graph_edges = [], [], [], []

    # ---------------------------------------------------------
    # 2. DETERMINISTIC DEDUPLICATION
    # ---------------------------------------------------------
    print(f"\n[Phase 2] Deterministic Cleanup...")
    df['norm_title'] = df['title'].apply(normalize_title)
    
    grouped = df.groupby('norm_title')
    unique_rows_phase1 = []
    det_removed_count = 0
    
    for _, group in tqdm(grouped, desc="Grouping Exact Matches", bar_format=BAR_FMT):
        records = group.to_dict('records')
        if len(records) > 1:
            survivor = resolve_cluster(records, reason="Exact Title")
            unique_rows_phase1.append(survivor)
            det_removed_count += (len(records) - 1)
            for rec in records:
                if rec['id'] != survivor['id']:
                    merge_log.append({
                        "survivor_id": survivor['id'], "survivor_title": survivor['title'],
                        "dropped_id": rec['id'], "dropped_title": rec['title'],
                        "reason": "Exact Match", "vector_score": 1.0, "jaccard_score": 1.0
                    })
        else:
            unique_rows_phase1.append(records[0])
            
    df_clean = pd.DataFrame(unique_rows_phase1)
    df_clean = df_clean.reset_index(drop=True)
    print(f"-> Deterministic Pass removed {det_removed_count:,} records.")

# ---------------------------------------------------------
# 2.5 TF-IDF TITLE VECTORS (for veto similarity)
# ---------------------------------------------------------
    print(f"\n[Phase 2.5] Calculating Anchor Weights (Noise Removal)...")
    
    # Pre-compute "Anchor Text" (Titles without stop words)
    df_clean['anchor_text'] = df_clean['title'].apply(get_anchor_text)
    
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word',
        min_df=1,              
        ngram_range=(1, 1),    
        use_idf=True,
        smooth_idf=True
    )
    
    print("-> Vectorizing Anchors...")
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_clean['anchor_text'])
    print(f"-> Anchor Vocab Size: {len(tfidf_vectorizer.get_feature_names_out()):,}")

    # ---------------------------------------------------------
    # 3. SEMANTIC EMBEDDING
    # ---------------------------------------------------------
    print(f"\n[Phase 3] Semantic Analysis...")
    embedder = SentenceTransformer(MODEL_NAME, device=device)
    
    embeddings = embedder.encode(
        df_clean['description'].tolist(), 
        batch_size=BATCH_SIZE, 
        show_progress_bar=True, 
        convert_to_tensor=True, 
        normalize_embeddings=True
    )

    # ---------------------------------------------------------
    # 4. HYBRID SEARCH (WITH SMART VETO)
    # ---------------------------------------------------------
    print(f"\n[Phase 4] Hybrid Graph Construction (Smart Veto)...")
    
    G = nx.Graph()
    G.add_nodes_from(range(len(df_clean)))
    
    all_ingredients = [parse_ingredients(x) for x in df_clean['products_extracted']]
    all_titles_norm = df_clean['norm_title'].tolist() # Use normalized for substring check
    all_ids = df_clean['id'].tolist()
    all_orig_titles = df_clean['title'].tolist()
    
    chunk_size = 256 
    scatter_counter = 0
    MAX_SCATTER_POINTS = 20000 
    
    if str(device) == "mps": embeddings = embeddings.to("mps")
    elif str(device) == "cuda": embeddings = embeddings.to("cuda")

    total_embeddings = len(embeddings)

    for i in tqdm(range(0, total_embeddings, chunk_size), desc="Semantic Search", bar_format=BAR_FMT):
        end_idx = min(i + chunk_size, total_embeddings)
        chunk_emb = embeddings[i : end_idx]
        
        batch_hits = util.semantic_search(chunk_emb, embeddings, top_k=SEARCH_K, score_function=util.dot_score)
        
        for local_idx, hits in enumerate(batch_hits):
            global_idx = i + local_idx
            vec_a = tfidf_matrix[global_idx]
            norm_a = all_titles_norm[global_idx] # For Substring check
            
            for hit in hits:
                neighbor_idx = hit['corpus_id']
                vector_score = hit['score']
                
                if neighbor_idx == global_idx: continue
                if vector_score < 0.6: continue
                
                # --- [SMART VETO LOGIC] ---
                is_safe_merge = False
                veto_reason = ""
                title_sim = 0.0
                
                # 1. Check Similarity only if potential merge
                if vector_score >= SIMILARITY_THRESHOLD:
                    vec_b = tfidf_matrix[neighbor_idx]
                    title_sim = (vec_a * vec_b.T).toarray()[0][0]
                    
                    if title_sim >= TITLE_SAFETY_THRESHOLD:
                        is_safe_merge = True
                    else:
                        # Similarity is LOW. Suspicious.
                        # Check Exception 1: Substring (e.g., "Musaka" inside "Musaka with potatoes")
                        norm_b = all_titles_norm[neighbor_idx]
                        if norm_a in norm_b or norm_b in norm_a:
                            is_safe_merge = True
                        else:
                            # Check Exception 2: Typo-ish similarity (SequenceMatcher ratio > 0.90)
                            lev_ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()
                            if lev_ratio > 0.90:
                                is_safe_merge = True
                            else:
                                # FINAL VETO
                                is_safe_merge = False
                                veto_reason = "ANCHOR_VETO"
                
                # If we passed semantic threshold but failed Veto -> Log and Skip
                if vector_score >= SIMILARITY_THRESHOLD and not is_safe_merge:
                    if random.random() < 0.2: # Log sample
                        near_miss_log.append({
                            "recipe_a": all_orig_titles[global_idx],
                            "recipe_b": all_orig_titles[neighbor_idx],
                            "reason": veto_reason,
                            "vector_score": float(vector_score),
                            "title_tfidf_sim": float(title_sim)
                        })
                    continue 

                # --- END VETO ---

                ing_b = all_ingredients[neighbor_idx]
                jaccard = calculate_jaccard(all_ingredients[global_idx], ing_b)
                
                # Logging
                if vector_score > 0.80:
                    if scatter_counter < MAX_SCATTER_POINTS and random.random() < 0.3:
                        scatter_log.append({
                            "vector_score": round(float(vector_score), 4),
                            "jaccard_score": round(jaccard, 4),
                            "title_sim": round(float(title_sim), 4),
                            "is_duplicate": (vector_score >= SIMILARITY_THRESHOLD and jaccard >= INGREDIENT_OVERLAP_THRESHOLD)
                        })
                        scatter_counter += 1

                # Final Decision
                if vector_score >= SIMILARITY_THRESHOLD:
                    if jaccard >= INGREDIENT_OVERLAP_THRESHOLD:
                        G.add_edge(global_idx, neighbor_idx)
                        if global_idx < neighbor_idx:
                            graph_edges.append({
                                "source": all_ids[global_idx],
                                "target": all_ids[neighbor_idx],
                                "weight": float(vector_score),
                                "type": "Undirected"
                            })
                    else:
                        # Near Miss (Ingredient Mismatch)
                        if global_idx < neighbor_idx:
                            near_miss_log.append({
                                "recipe_a": all_orig_titles[global_idx],
                                "recipe_b": all_orig_titles[neighbor_idx],
                                "reason": "INGREDIENT_MISMATCH",
                                "vector_score": float(vector_score),
                                "jaccard_score": jaccard
                            })

    # ---------------------------------------------------------
    # 5. RESOLUTION
    # ---------------------------------------------------------
    print(f"\n[Phase 5] Resolving Clusters...")
    clusters = list(nx.connected_components(G))
    final_records = []
    df_clean_records = df_clean.to_dict('records')
    semantic_dupes_count = 0
    
    for cluster in tqdm(clusters, desc="Consolidating", bar_format=BAR_FMT):
        indices = list(cluster)
        if len(indices) > 1:
            candidates = [df_clean_records[i] for i in indices]
            survivor = resolve_cluster(candidates, reason="Semantic Cluster")
            final_records.append(survivor)
            semantic_dupes_count += (len(indices) - 1)
            for cand in candidates:
                if cand['id'] != survivor['id']:
                    merge_log.append({
                        "survivor_id": survivor['id'], "survivor_title": survivor['title'],
                        "dropped_id": cand['id'], "dropped_title": cand['title'],
                        "reason": "Semantic+Jaccard", "vector_score": "High", "jaccard_score": "High"
                    })
        else:
            final_records.append(df_clean_records[indices[0]])

    # ---------------------------------------------------------
    # 6. EXPORT
    # ---------------------------------------------------------
    print(f"\n[Phase 6] Writing Output...")
    df_final = pd.DataFrame(final_records)
    drop_cols = [c for c in ['norm_title', 'anchor_text'] if c in df_final.columns]
    df_final.drop(columns=drop_cols, inplace=True)
        
    df_final.to_csv(OUTPUT_FILE, sep='\t', index=False)
    pd.DataFrame(merge_log).to_csv(REPORT_MERGES, index=False)
    nm_df = pd.DataFrame(near_miss_log)
    if not nm_df.empty: nm_df.to_csv(REPORT_NEAR_MISSES, index=False)
    pd.DataFrame(scatter_log).to_csv(REPORT_SCATTER, index=False)
    pd.DataFrame(graph_edges).to_csv(REPORT_GRAPH_EDGES, index=False)

    print("\n" + "="*60)
    print(f"DONE. Final Unique Count: {len(df_final):,}")
    print(f"Removed via Deterministic: {det_removed_count:,}")
    print(f"Removed via Semantic: {semantic_dupes_count:,}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
