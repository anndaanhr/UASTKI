# -*- coding: utf-8 -*-
"""
Search Engine Module — Hybrid TF-IDF + Sentence-BERT
Sistem pencarian hybrid yang menggabungkan TF-IDF dan Sentence-BERT
untuk meningkatkan akurasi pencarian dokumen berita Timur Tengah.

Mode operasi:
  - PRECOMPUTED (deploy): Load artefak dari file .pkl/.npz/.npy
  - COMPUTE (lokal): Hitung TF-IDF & BERT dari awal (fallback)
"""

import numpy as np
import os
import gc
import pickle

# Graceful torch import — required by sentence-transformers
try:
    import torch
    # Fallback for compatibility: newer transformers versions expect torch.float8_e8m0fnu
    if not hasattr(torch, "float8_e8m0fnu"):
        setattr(torch, "float8_e8m0fnu", torch.float32)
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] torch not available — BERT scoring disabled, TF-IDF only mode.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from dataset import clean_text

BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
W_TFIDF = 0.35
W_BERT  = 0.65

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFIDF_VEC_PATH    = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.npz")
BERT_CACHE_PATH   = os.path.join(BASE_DIR, "bert_embeddings.npy")


class HybridSearchEngine:
    def __init__(self, df):
        """
        Inisialisasi Hybrid Search Engine (TF-IDF + Sentence-BERT).
        Otomatis mendeteksi apakah artefak precomputed tersedia.

        :param df: DataFrame yang sudah memiliki kolom 'processed_text' dan 'content'.
        """
        self.df = df.copy()
        self.bert_model = None
        self.bert_emb = None

        if TORCH_AVAILABLE:
            import torch as _torch
            self.device = "cuda" if _torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        print(f"[ENGINE] Device: {self.device.upper()}")

        if 'content' in self.df.columns and 'content_raw' not in self.df.columns:
            self.df = self.df.rename(columns={'content': 'content_raw'})
        elif 'content_raw' not in self.df.columns:
            self.df['content_raw'] = ''

        # === TF-IDF: Load precomputed or compute ===
        self._init_tfidf()

        # === BERT: Load precomputed embeddings ===
        self._init_bert()

        # Simpan dokumen sebagai list of dicts untuk akses cepat
        self.documents = self.df.to_dict('records')
        print("[ENGINE] Hybrid Search Engine siap!")

    def _init_tfidf(self):
        """Load TF-IDF dari file precomputed, atau hitung dari awal."""
        if os.path.exists(TFIDF_VEC_PATH) and os.path.exists(TFIDF_MATRIX_PATH):
            # === PRECOMPUTED MODE ===
            print(f"[ENGINE] Memuat TF-IDF vectorizer dari: {TFIDF_VEC_PATH}")
            with open(TFIDF_VEC_PATH, "rb") as f:
                self.tfidf_vec = pickle.load(f)

            print(f"[ENGINE] Memuat TF-IDF matrix dari: {TFIDF_MATRIX_PATH}")
            self.tfidf_matrix = sparse.load_npz(TFIDF_MATRIX_PATH)
            print(f"[ENGINE] TF-IDF matrix shape: {self.tfidf_matrix.shape}")

            # Validasi: jumlah baris harus sesuai dengan jumlah dokumen
            if self.tfidf_matrix.shape[0] != len(self.df):
                print(f"[ENGINE][WARNING] TF-IDF matrix rows ({self.tfidf_matrix.shape[0]}) "
                      f"!= documents ({len(self.df)}). Recomputing...")
                self._compute_tfidf()
        else:
            # === COMPUTE MODE (fallback) ===
            self._compute_tfidf()

    def _compute_tfidf(self):
        """Hitung TF-IDF dari awal (mode lokal/fallback)."""
        print("[ENGINE] Membangun indeks TF-IDF ...")
        self.texts = self.df['processed_text'].fillna("").tolist()
        self.tfidf_vec = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf_vec.fit_transform(self.texts)
        print(f"[ENGINE] TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        gc.collect()

    def _init_bert(self):
        """Load BERT embeddings precomputed. Model di-load lazy saat query pertama."""
        # Load precomputed embeddings jika ada
        if os.path.exists(BERT_CACHE_PATH):
            cached = np.load(BERT_CACHE_PATH)
            if cached.shape[0] == len(self.df):
                print(f"[ENGINE] Memuat BERT embeddings dari cache: {BERT_CACHE_PATH}")
                self.bert_emb = cached
            else:
                print(f"[ENGINE][WARNING] BERT cache mismatch ({cached.shape[0]} vs {len(self.df)})")
                self.bert_emb = None
        else:
            print("[ENGINE] BERT embeddings cache tidak ditemukan.")
            self.bert_emb = None

        # BERT model di-load lazy — TIDAK saat startup
        # Ini menghemat RAM karena model hanya di-load saat ada query pertama
        self.bert_model = None  # akan di-load di _get_bert_model()
        gc.collect()

    def _get_bert_model(self):
        """Lazy-load BERT model saat pertama kali dibutuhkan."""
        if self.bert_model is None and TORCH_AVAILABLE and self.bert_emb is not None:
            try:
                from sentence_transformers import SentenceTransformer
                print("[ENGINE] Lazy-loading Sentence-BERT model untuk query encoding ...")
                self.bert_model = SentenceTransformer(BERT_MODEL, device=self.device)
                print("[ENGINE] Sentence-BERT model siap!")
            except Exception as e:
                print(f"[ENGINE][WARNING] Gagal load BERT model: {e}")
                self.bert_model = None
        return self.bert_model

    def search(self, query, top_k=10):
        """
        Melakukan pencarian hybrid: TF-IDF + Sentence-BERT.

        :param query: String kueri pencarian.
        :param top_k: Jumlah hasil maksimal yang dikembalikan.
        :return: Dict berisi results list, query_tokens, dll.
        """
        query_proc = clean_text(query)

        # Jika kueri kosong setelah preprocessing
        if not query_proc.strip():
            return {"results": [], "query_tokens": []}

        query_tokens = query_proc.split()

        # === TF-IDF Score ===
        q_tfidf = self.tfidf_vec.transform([query_proc])
        tfidf_scores = cosine_similarity(q_tfidf, self.tfidf_matrix).flatten()

        # === BERT Score (lazy-load model jika belum ada) ===
        bert_model = self._get_bert_model()
        if bert_model is not None and self.bert_emb is not None:
            q_bert = bert_model.encode(
                [query], normalize_embeddings=True, device=self.device
            )
            bert_scores = np.dot(self.bert_emb, q_bert.T).flatten()
            bert_scores = (bert_scores + 1) / 2  # Normalize dari [-1,1] ke [0,1]

            # === Normalisasi & Gabungkan ===
            if tfidf_scores.max() > 0:
                tfidf_scores = tfidf_scores / tfidf_scores.max()
            combined = W_TFIDF * tfidf_scores + W_BERT * bert_scores
        else:
            # TF-IDF only fallback
            if tfidf_scores.max() > 0:
                tfidf_scores = tfidf_scores / tfidf_scores.max()
            combined = tfidf_scores
            bert_scores = np.zeros_like(tfidf_scores)

        top_idx = np.argsort(combined)[::-1][:top_k]

        results_list = []
        for rank, i in enumerate(top_idx):
            doc = self.documents[i]
            results_list.append({
                "rank": rank + 1,
                "title": doc.get('title', ''),
                "relevance_score": float(combined[i]),
                "tfidf_score": float(tfidf_scores[i]),
                "bert_score": float(bert_scores[i]),
                "url": doc.get('url', ''),
                "source": doc.get('source', 'News'),
                "author": doc.get('author', 'Unknown'),
                "date": doc.get('date', 'Unknown'),
                "license": doc.get('license', ''),
                "text": doc.get('content_raw', '')[:1000],
                "doc_id": doc.get('doc_id', ''),
            })

        return {
            "results": results_list,
            "query_tokens": query_tokens,
        }


if __name__ == "__main__":
    from dataset import load_and_preprocess

    # Unit Test untuk memastikan Hybrid Engine bekerja
    print("=== PENGUJIAN SEARCH ENGINE HYBRID TF-IDF + BERT ===")

    # Gunakan sampel data untuk pengujian lokal agar lebih cepat
    df = load_and_preprocess(sample_size=100)

    engine = HybridSearchEngine(df)

    queries_to_test = [
        "Iran nuclear deal negotiations",
        "Middle East revolution political crisis",
        "Turkey Erdogan democracy politics",
    ]

    for q in queries_to_test:
        print(f"\nKueri: '{q}'")
        search_data = engine.search(q, top_k=5)
        results = search_data['results']

        print(f"{'#':<4} {'Score':>7} {'TF-IDF':>8} {'BERT':>7}  Judul")
        print("-" * 75)
        for r in results:
            print(f"{r['rank']:<4} {r['relevance_score']:>7.4f} "
                  f"{r['tfidf_score']:>8.4f} {r['bert_score']:>7.4f}  "
                  f"{r['title'][:50]}")
