# -*- coding: utf-8 -*-
"""
Search Engine Module — Hybrid TF-IDF + Sentence-BERT
Sistem pencarian hybrid yang menggabungkan TF-IDF dan Sentence-BERT
untuk meningkatkan akurasi pencarian dokumen berita Timur Tengah.
"""

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dataset import clean_text

# =========================================================
# KONFIGURASI
# =========================================================
BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
W_TFIDF = 0.35
W_BERT = 0.65


class HybridSearchEngine:
    def __init__(self, df):
        """
        Inisialisasi Hybrid Search Engine (TF-IDF + Sentence-BERT).

        :param df: DataFrame yang sudah memiliki kolom 'processed_text' dan 'content'.
        """
        self.df = df.copy()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ENGINE] Device: {self.device.upper()}")

        # Pastikan kolom content tersedia (rename jika perlu)
        if 'content' in self.df.columns and 'content_raw' not in self.df.columns:
            self.df = self.df.rename(columns={'content': 'content_raw'})
        elif 'content_raw' not in self.df.columns:
            self.df['content_raw'] = ''

        # === TF-IDF Indexing ===
        print("[ENGINE] Membangun indeks TF-IDF ...")
        self.texts = self.df['processed_text'].fillna("").tolist()
        self.tfidf_vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf_vec.fit_transform(self.texts)
        print(f"[ENGINE] TF-IDF matrix shape: {self.tfidf_matrix.shape}")

        # === BERT Embeddings ===
        print("[ENGINE] Memuat model Sentence-BERT ...")
        self.bert_model = SentenceTransformer(BERT_MODEL, device=self.device)

        print("[ENGINE] Menghitung BERT embeddings untuk seluruh corpus ...")
        semantic_inputs = (
            self.df['title'].fillna('') + ". " +
            self.df['content_raw'].fillna('').str[:500]
        ).tolist()

        self.bert_emb = self.bert_model.encode(
            semantic_inputs,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print(f"[ENGINE] BERT embeddings shape: {self.bert_emb.shape}")

        # Simpan dokumen sebagai list of dicts untuk akses cepat
        self.documents = self.df.to_dict('records')
        print("[ENGINE] Hybrid Search Engine siap!")

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

        # === BERT Score ===
        q_bert = self.bert_model.encode(
            [query], normalize_embeddings=True, device=self.device
        )
        bert_scores = np.dot(self.bert_emb, q_bert.T).flatten()

        # === Normalisasi & Gabungkan ===
        if tfidf_scores.max() > 0:
            tfidf_scores = tfidf_scores / tfidf_scores.max()
        bert_scores = (bert_scores + 1) / 2  # Normalize dari [-1,1] ke [0,1]

        combined = W_TFIDF * tfidf_scores + W_BERT * bert_scores
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
