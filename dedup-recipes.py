import sys
import subprocess
import importlib
import os
import json
import time
import random

# =========================
# FIX FOR MACOS M
# =========================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ========================
# 1. AUTO-INSTALL PREREQUISITES & NUMPY FIX
# =========================
REQUIRED_PACKAGES = [
    "torch", 
    "sentence-transformers", 
    "faiss-cpu", 
    "pandas", 
    "networkx", 
    "tqdm"
]

def install_packages():
    print("System Check: Verifying dependencies...")
    try:
        import numpy
        if int(numpy.__version__.split('.')[0]) >= 2:
            print(f"Detected NumPy {numpy.__version__} (Incompatible). Downgrading...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2", "-q"])
            print("NumPy downgraded. Please restart the script.")
            sys.exit(0)
    except ImportError:
        pass 

    needs_install = []
    for package in REQUIRED_PACKAGES:
        pkg_import_name = package.replace("-", "_").split("=")[0]
        if pkg_import_name == "faiss_cpu": pkg_import_name = "faiss"
        try:
            importlib.import_module(pkg_import_name)
        except ImportError:
            needs_install.append(package)
    
    if needs_install:
        print(f"Installing missing packages: {', '.join(needs_install)}...")
        for pkg in needs_install:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        print("Dependencies installed.\n")
    else:
        print("All dependencies are present.\n")

install_packages()

# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import torch
import faiss
import networkx as nx
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =========================
# CONFIGS
# =========================
INPUT_FILE = "bg-recipes-final.jsonl"
OUTPUT_JSONL = "bg-recipes-deduplicated.jsonl"
OUTPUT_TSV = "bg-recipes-deduplicated.tsv"

# LOGGING
DEBUG_MODE = True        
LOG_SAMPLE_RATE = 0.05   

# TUNING PARAMETERS
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SIMILARITY_THRESHOLD = 0.96      
INGREDIENT_OVERLAP_THRESHOLD = 0.50 
BATCH_SIZE = 128             

BAR_FMT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

# =========================
# LOGIC
# =========================

def setup_device():
    if torch.backends.mps.is_available():
        print("HARDWARE ACCELERATION: Apple M-Series GPU (MPS) active.")
        return "mps"
    elif torch.cuda.is_available():
        print("HARDWARE ACCELERATION: NVIDIA CUDA active.")
        return "cuda"
    else:
        print("WARNING: No Hardware Acceleration detected. Running slowly on CPU.")
        return "cpu"

def normalize_title(text):
    if not isinstance(text, str): return ""
    clean = "".join(c for c in text.lower() if c.isalnum() or c.isspace())
    return " ".join(clean.split())

def calculate_ingredient_overlap(list_a, list_b):
    """
    Returns Jaccard Similarity (0.0 to 1.0) of two ingredient lists.
    """
    if not list_a or not list_b:
        return 1.0 # Benefit of doubt for empty data
        
    set_a = set(x.lower().strip() for x in list_a)
    set_b = set(x.lower().strip() for x in list_b)
    
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    if union == 0: return 0.0
    return intersection / union

def resolve_cluster(cluster_rows, reason="Unknown"):
    # Updated key: 'products_extracted' instead of 'extracted_ingredients'
    survivor = sorted(
        cluster_rows, 
        key=lambda x: (
            x.get('quality_flag', '') == 'ok',  # Updated logic
            len(x.get('description', '')), 
            len(x.get('products_extracted', []))
        ), 
        reverse=True
    )[0]
    
    if DEBUG_MODE and len(cluster_rows) > 1 and random.random() < LOG_SAMPLE_RATE:
        titles = [r.get('title', 'Unknown')[:30] for r in cluster_rows]
        tqdm.write(f"[DEBUG] Merged {len(cluster_rows)} ({reason}): {titles} -> '{survivor.get('title')}'")
    
    return survivor

def main():
    print("="*50)
    print("RECIPE DEDUPLICATION ENGINE (HYBRID v2.1)")
    print("="*50 + "\n")
    
    faiss.omp_set_num_threads(1) 
    device = setup_device()
    
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"\n[Phase 1] Loading Dataset...")
    df = pd.read_json(INPUT_FILE, lines=True)
    initial_count = len(df)
    print(f"Loaded {initial_count:,} recipes.")

    # 2. Deterministic Deduplication
    print(f"\n[Phase 2] Deterministic Cleanup (Titles)")
    df['norm_title'] = df['title'].apply(normalize_title)
    
    grouped = df.groupby('norm_title')
    unique_rows = []
    
    for _, group in tqdm(grouped, total=len(grouped), desc="Processing Groups", bar_format=BAR_FMT):
        records = group.to_dict('records')
        survivor = resolve_cluster(records, reason="Exact Title")
        unique_rows.append(survivor)
    
    df_clean = pd.DataFrame(unique_rows)
    det_removed = initial_count - len(df_clean)
    print(f"Removed {det_removed:,} exact duplicates.")

    # 3. Semantic Deduplication
    print(f"\n[Phase 3] Semantic Analysis (Neural Engine)")
    print(f"Loading Model: {MODEL_NAME}...")
    
    embedder = SentenceTransformer(MODEL_NAME, device=device)
    descriptions = df_clean['description'].fillna("").tolist()
    
    print("Vectorizing Descriptions...")
    embeddings = embedder.encode(
        descriptions, 
        batch_size=BATCH_SIZE, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # 4. Indexing & Hybrid Clustering
    print(f"\n[Phase 4] Hybrid Search (Vector + Ingredients)")
    print(f"   -> Vector Threshold: > {SIMILARITY_THRESHOLD}")
    print(f"   -> Ingredient Overlap: > {INGREDIENT_OVERLAP_THRESHOLD*100}%")
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d) 
    index.add(embeddings)
    
    G = nx.Graph()
    G.add_nodes_from(range(len(df_clean)))
    
    chunk_size = 500
    
    # FIX: Use correct column name 'products_extracted'
    all_ingredients = df_clean['products_extracted'].tolist()
    idx_to_title = df_clean['title'].to_dict()

    for i in tqdm(range(0, len(embeddings), chunk_size), desc="Connecting Neighbors", bar_format=BAR_FMT):
        chunk_emb = embeddings[i : i + chunk_size]
        D, I = index.search(chunk_emb, k=5) 
        
        for local_idx in range(len(chunk_emb)):
            global_idx = i + local_idx
            
            # Get ingredients for current item
            ing_a = all_ingredients[global_idx]
            
            for rank, neighbor_idx in enumerate(I[local_idx]):
                if neighbor_idx == -1 or neighbor_idx == global_idx:
                    continue
                
                vector_score = D[local_idx][rank]
                
                if vector_score >= SIMILARITY_THRESHOLD:
                    ing_b = all_ingredients[neighbor_idx]
                    
                    overlap = calculate_ingredient_overlap(ing_a, ing_b)
                    
                    if overlap >= INGREDIENT_OVERLAP_THRESHOLD:
                        G.add_edge(global_idx, neighbor_idx)
                        
                        if DEBUG_MODE and random.random() < 0.01:
                            t1 = idx_to_title.get(global_idx, "Unknown")[:20]
                            t2 = idx_to_title.get(neighbor_idx, "Unknown")[:20]
                            tqdm.write(f"[MATCH] {vector_score:.3f} | Ingr: {overlap*100:.0f}% | {t1} == {t2}")

    # 5. Resolution
    print(f"\n[Phase 5] Resolving Conflicts")
    clusters = list(nx.connected_components(G))
    final_records = []
    df_records = df_clean.to_dict('records')
    semantic_dupes = 0
    
    for cluster in tqdm(clusters, desc="Finalizing Dataset", bar_format=BAR_FMT):
        indices = list(cluster)
        if len(indices) > 1:
            semantic_dupes += (len(indices) - 1)
            candidates = [df_records[i] for i in indices]
            survivor = resolve_cluster(candidates, reason="Hybrid Cluster")
            final_records.append(survivor)
        else:
            final_records.append(df_records[indices[0]])

    # 6. Saving
    print(f"\n[Phase 6] Saving Final Datasets")
    
    print(f"Writing JSONL to {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for rec in tqdm(final_records, desc="Writing Lines", unit="row", bar_format=BAR_FMT):
            if 'norm_title' in rec: del rec['norm_title']
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Writing TSV to {OUTPUT_TSV}...")
    df_final = pd.DataFrame(final_records)
    
    # FIX: Use correct column name here as well
    if 'products_extracted' in df_final.columns:
        df_final['products_extracted'] = df_final['products_extracted'].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )
    
    if 'norm_title' in df_final.columns:
        df_final.drop(columns=['norm_title'], inplace=True)
        
    df_final.to_csv(OUTPUT_TSV, sep='\t', index=False)

    print("\n" + "="*50)
    print("FINAL REPORT (HYBRID)")
    print("="*50)
    print(f"Original Recipes    : {initial_count:,}")
    print(f"Exact Duplicates    : -{det_removed:,}")
    print(f"Semantic Duplicates : -{semantic_dupes:,}")
    print(f"-----------------------------")
    print(f"Final Unique Count  : {len(final_records):,}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()