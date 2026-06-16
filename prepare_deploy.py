# -*- coding: utf-8 -*-
"""
Prepare Deploy — Precompute semua artefak ML di laptop sebelum deploy.

Jalankan sekali di laptop:
    python prepare_deploy.py

Artefak yang dihasilkan:
    - corpus_preprocessed.csv  (dataset cache)
    - tfidf_vectorizer.pkl     (fitted TF-IDF vectorizer)
    - tfidf_matrix.npz         (sparse TF-IDF matrix)
    - bert_embeddings.npy      (BERT embeddings untuk seluruh corpus)

Semua file ini akan di-upload saat `railway up`, sehingga server
hanya perlu LOAD file, bukan COMPUTE dari awal.
"""

import sys
import os
import pickle
import gc

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TFIDF_VEC_PATH    = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.npz")
BERT_CACHE_PATH   = os.path.join(BASE_DIR, "bert_embeddings.npy")

# ============================================================
# STEP 1: Dataset download & preprocessing
# ============================================================
from dataset import download_dataset, load_and_preprocess

print("=" * 60)
print("STEP 1/3: Preparing dataset ...")
print("=" * 60)
df = download_dataset()
print(f"  Raw dataset  : {len(df):,} dokumen")

df_proc = load_and_preprocess()
print(f"  Preprocessed : {len(df_proc):,} dokumen")

# ============================================================
# STEP 2: Build & save TF-IDF
# ============================================================
print()
print("=" * 60)
print("STEP 2/3: Building TF-IDF index ...")
print("=" * 60)

texts = df_proc['processed_text'].fillna("").tolist()
tfidf_vec = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))
tfidf_matrix = tfidf_vec.fit_transform(texts)
print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")

# Save vectorizer
with open(TFIDF_VEC_PATH, "wb") as f:
    pickle.dump(tfidf_vec, f)
print(f"  Saved: {TFIDF_VEC_PATH}")

# Save sparse matrix
sparse.save_npz(TFIDF_MATRIX_PATH, tfidf_matrix)
print(f"  Saved: {TFIDF_MATRIX_PATH}")

gc.collect()

# ============================================================
# STEP 3: Compute & save BERT embeddings
# ============================================================
print()
print("=" * 60)
print("STEP 3/3: Computing BERT embeddings ...")
print("=" * 60)

try:
    import torch
    if not hasattr(torch, "float8_e8m0fnu"):
        setattr(torch, "float8_e8m0fnu", torch.float32)

    from sentence_transformers import SentenceTransformer

    BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device.upper()}")

    model = SentenceTransformer(BERT_MODEL, device=device)

    # Rename 'content' to 'content_raw' if needed for input
    if 'content' in df_proc.columns:
        content_col = 'content'
    elif 'content_raw' in df_proc.columns:
        content_col = 'content_raw'
    else:
        content_col = None

    if content_col:
        semantic_inputs = (
            df_proc['title'].fillna('') + ". " +
            df_proc[content_col].fillna('').str[:300]
        ).tolist()
    else:
        semantic_inputs = df_proc['title'].fillna('').tolist()

    bert_emb = model.encode(
        semantic_inputs,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  BERT embeddings shape: {bert_emb.shape}")

    np.save(BERT_CACHE_PATH, bert_emb)
    print(f"  Saved: {BERT_CACHE_PATH}")

except Exception as e:
    print(f"  [WARNING] BERT computation failed: {e}")
    print(f"  Skipping BERT — TF-IDF only mode will be used.")

# ============================================================
print()
print("=" * 60)
print("DONE! Files ready for deployment:")
print("=" * 60)
for fpath in [TFIDF_VEC_PATH, TFIDF_MATRIX_PATH, BERT_CACHE_PATH]:
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  ✓ {os.path.basename(fpath):30s} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ {os.path.basename(fpath):30s} (not created)")

print()
print("Now run: railway up")
