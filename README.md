# Hybrid Deduplication Engine (The "Anchor Veto" Approach)

### Why this is not “just" cosine similarity?

This repository demonstrates a production-grade approach to cleaning noisy, real-world datasets.

Although I built it to prevent a diplomatic incident involving Balkan sausages (read the [article](https://www.linkedin.com/in/damyandeschev/) for the backstory), this is **NOT** just a recipe tool.

The core problem I solve here is: **How to deduplicate semantically similar records without destroying meaningful variation?**

If your domain is job ads, legal clauses, medical records, or e-commerce products, the same principles apply.

---

## The Problem: Why "Embeddings" are not a magic wand

In theory, `Embeddings + Vector DB + Cosine Similarity` should be enough to clean data.
In practice, that pipeline is a "hallucination machine."

Real datasets suffer from:
* **The "Vibe" Problem:** Vector search finds things that *feel* related, not things that *are* the same.
* **The "Boilerplate" Trap:** Two legal contracts might share 90% identical boilerplate text but differ in the one clause that matters ("Termination for Cause" vs "Termination for Convenience"). Vector search sees 90% overlap and merges them.
* **The "Omelette" Fallacy:** If you compare raw features (e.g., ingredients), a Chocolate Cake and a Quiche look identical (Flour, Eggs, Oil, Salt).

If I deduplicated purely on vector similarity, I would have collapsed distinct entities into a single, corrupted record.

---

## The Solution: A Hybrid "Veto" Architecture

I do not trust the AI blindly. I use a layered "Check and Balance" system.

### Phase 1: The Boring Stuff (Deterministic)
Before firing up the GPUs, I do standard string normalization.
* **Logic:** `lower()`, `strip()`, remove special chars.
* **Why:** It is faster and safer to merge "Chicken Rice" and "chicken-rice" using code than using a neural network.

### Phase 2: Candidate Generation (Semantic)
I use a Sentence Transformer (default: `paraphrase-multilingual-MiniLM-L12-v2`) to create embeddings and find candidates with high cosine similarity (`>0.94`).
* **Role:** This finds the "Hidden" duplicates (e.g., "Roasted Bird" vs "Chicken in Oven").
* **Status:** These are just *candidates*, not confirmed merges.

### Phase 3: The "Anchor Veto" (The Guardrail)
**This is the most important part.**
Even if the Vector Model says two records are 99% similar, I run a "Veto" check on the Title/Header.
* **The Logic:** I strip "noise" words (prepositions, conjunctions) from the titles and run a TF-IDF comparison on the remaining "Anchor" nouns.
* **The Rule:** If the semantic vectors match, but the **Identity Anchors** (the nouns) do not, I block the merge.
* *Example:* This stops the model from merging "Lukanka" and "Sudzhuk" (two different sausages that look identical to a vector model).

### Phase 4: Domain Verification (Jaccard)
I verify the "substance" of the record.
* **The Logic:** In this repo, I compare ingredient lists using Jaccard Similarity.
* **The Twist:** I strip ubiquitous ingredients (Salt, Water, Oil) before comparing. This forces the model to compare the *soul* of the item, not its chemistry set.

---

## The Graph Logic (Cluster Resolution)

Duplicates don't come in pairs; they come in messy clusters.
Instead of pairwise deletion (which is order-dependent and buggy), I build a **NetworkX Graph**:

1.  **Nodes:** All records.
2.  **Edges:** "Safe" merges confirmed by the 4-phase logic above.
3.  **Resolution:** I find connected components (clusters) and collapse them into a single survivor.

**The Survivor Policy:**
I don't pick at random. The script keeps the record that is:
1.  Longest (most descriptive).
2.  Most "complete" (highest number of structured features).

---

## Generalizing this beyond Recipes

To adapt this pipeline to your job (Fintech, Legal, E-commerce), change these three functions in `dedup_demonstrator.py`:

1.  **The "Anchor" (`get_anchor_text`):**
    * *Recipes:* Title minus "with", "and", "in".
    * *E-commerce:* Product Name minus "Pro", "Max", "2024", "Edition".
    * *Legal:* Clause Header minus "Section", "Paragraph".

2.  **The "Substance" (`products_extracted`):**
    * *Recipes:* Ingredients.
    * *HR:* Skills list (Python, SQL, Management).
    * *Retail:* Specs (Size, Color, Material).

3.  **The Stop Lists (`STOP_INGREDIENTS`):**
    * Remove the high-frequency noise specific to your industry (e.g., "manager" in job titles, "inc." in company names).

---

## How to use

The script `dedup_demonstrator.py` is self-contained.

1.  **Prerequisites:** Python 3.8+.
2.  **Install:** The script detects missing packages (`torch`, `sentence-transformers`, `networkx`, `pandas`) and auto-installs them.
3.  **Hardware:** It automatically detects CUDA (Nvidia) or MPS (Mac Silicon) for acceleration.
4.  **Run:**
    ```bash
    python dedup_demonstrator.py
    ```
5.  **Output:**
    * `bg-recipes-deduplicated_final.tsv`: The clean dataset.
    * `report_merges.csv`: A log of exactly what was merged and why (auditable).
    * `report_near_misses.csv`: A log of what the AI *wanted* to merge but the "Veto" stopped (entertaining reading).

---

## Conceptual Summary

* **Not a classifier.**
* **Not a black box.**
* **Not "AGI cleaning your data".**

It is a probabilistic engine with deterministic brakes. I favor **Precision over Recall**: in my case it is better to leave two duplicates in the database than to accidentally merge entities because they sit in the same vector neighborhood. If your case differs, feel free to adapt the logic to your needs. 
