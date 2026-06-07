# -*- coding: utf-8 -*-
"""
Evaluation Module — Keyword-Based Relevance Evaluation
Evaluasi sistem pencarian hybrid dengan metrik P@K, R@K, dan MRR.
Termasuk tracking ke DagsHub MLflow.
"""

import sys
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# =========================================================
# GROUND TRUTH: KEYWORD-BASED EVALUATION QUERIES
# Konteks: Berita Timur Tengah
# =========================================================
EVAL_QUERIES = [
    {
        "query": "Iran nuclear deal negotiations",
        "keywords": ["iran", "nuclear", "deal", "negotiations", "agreement"],
    },
    {
        "query": "Middle East revolution political crisis",
        "keywords": ["revolution", "crisis", "political", "middle east", "egypt", "syria"],
    },
    {
        "query": "Turkey Erdogan democracy politics",
        "keywords": ["turkey", "erdogan", "democratic", "court", "politics", "ban"],
    },
    {
        "query": "Palestine Israel conflict peace",
        "keywords": ["palestine", "israel", "conflict", "peace", "trusteeship", "normalization"],
    },
    {
        "query": "US foreign policy Syria opposition",
        "keywords": ["us", "syria", "opposition", "policy", "military", "fighting"],
    },
]

K = 10  # Evaluasi pada top-K hasil

# =========================================================
# GROUND TRUTH: TITLE-BASED (untuk evaluasi presisi tinggi)
# =========================================================
GROUND_TRUTH = {
    "semantic web policies": [
        "Verification, Validation and Integrity of Distributed and Interchanged Rule Based Policies and Contracts in the Semantic Web"
    ],
    "electronic government in saudi arabia": [
        "Electronic-government in Saudi Arabia: A positive revolution in the peninsula",
        "EGovernment Stage Model: Evaluating the Rate of Web Development Progress of Government Websites in Saudi Arabia",
        "Success Factors Contributing to eGovernment Adoption in Saudi Arabia: G2C approach"
    ],
    "online retailing saudi arabia": [
        "A Conceptual Framework for the Promotion of Trusted Online Retailing Environment in Saudi Arabia",
        "Factors unflinching e-commerce adoption by retailers in Saudi Arabia: Qual Analysis",
        "Factors influencing E-commerce Adoption by Retailers in Saudi Arabia"
    ],
    "scientific communication kyrgyz republic": [
        "Toward a New Policy for Scientific and Technical Communication: the Case of Kyrgyz Republic"
    ],
    "middle east revolutions virtual communities": [
        "Virtual communities? the middle east revolutions at the Guardian forum: Comment Is Free"
    ],
    "b2c e-commerce saudi arabia": [
        "Wheel of B2C E-commerce Development in Saudi Arabia"
    ],
    "covid 19 cases saudi arabia": [
        "Google Searches and COVID-19 Cases in Saudi Arabia: A Correlation Study",
        "Weather impact on daily cases of COVID-19 in Saudi Arabia using machine learning"
    ],
    "machine learning public policy": [
        "Explainable Machine Learning for Public Policy: Use Cases, Gaps, and Research Directions"
    ],
    "scientific mobility middle east": [
        "Analysing Scientific Mobility and Collaboration in the Middle East and North Africa"
    ],
    "m-commerce saudi arabia private sector": [
        "Critical success factors for m-commerce in Saudi Arabia's private sector -- a multiple case study analysis"
    ]
}


# =========================================================
# FUNGSI METRIK EVALUASI
# =========================================================

def precision_at_k(relevance_list, k):
    """Menghitung Precision@K dari list boolean relevansi."""
    return sum(relevance_list[:k]) / k if k > 0 else 0.0


def recall_at_k(relevance_list, total_relevant, k):
    """Menghitung Recall@K dari list boolean relevansi."""
    hits = sum(relevance_list[:k])
    return hits / min(total_relevant, k) if total_relevant > 0 else 0.0


def reciprocal_rank(relevance_list):
    """Menghitung Reciprocal Rank (RR) — posisi pertama dokumen relevan."""
    for i, v in enumerate(relevance_list):
        if v:
            return 1.0 / (i + 1)
    return 0.0


# =========================================================
# FUNGSI EVALUASI UTAMA
# =========================================================

def evaluate_system(engine):
    """
    Evaluasi Search Engine terhadap EVAL_QUERIES menggunakan
    keyword-based relevance judgment.

    :param engine: Instance HybridSearchEngine.
    :return: (summary_dict, detailed_results_list)
    """
    documents = engine.documents

    eval_results = []

    for q in EVAL_QUERIES:
        search_data = engine.search(q['query'], top_k=K)
        results = search_data['results']
        kw = q['keywords']

        # Tentukan relevansi setiap hasil berdasarkan keyword matching
        relevance = []
        for r in results:
            doc_text = (r.get('title', '') + " " + r.get('text', '')).lower()
            is_relevant = any(kw_i.lower() in doc_text for kw_i in kw)
            relevance.append(is_relevant)

        # Hitung total dokumen relevan di seluruh corpus
        total_rel = max(sum(
            any(
                kw_i.lower() in (
                    d.get('title', '') + " " + d.get('content_raw', d.get('content', ''))
                ).lower()
                for kw_i in kw
            )
            for d in documents
        ), 1)

        hits = sum(relevance)
        p_k = precision_at_k(relevance, K)
        r_k = recall_at_k(relevance, total_rel, K)
        rr = reciprocal_rank(relevance)

        eval_results.append({
            "query": q['query'],
            "P@K": round(p_k, 4),
            "R@K": round(r_k, 4),
            "RR": round(rr, 4),
            "Hits": hits,
        })

    # Hitung rata-rata
    avg_pk = float(np.mean([r['P@K'] for r in eval_results]))
    avg_rk = float(np.mean([r['R@K'] for r in eval_results]))
    mrr = float(np.mean([r['RR'] for r in eval_results]))

    summary = {
        "Avg_P@K": round(avg_pk, 4),
        "Avg_R@K": round(avg_rk, 4),
        "MRR": round(mrr, 4),
        "K": K,
    }

    return summary, eval_results


# =========================================================
# MAIN: EVALUASI + MLFLOW TRACKING
# =========================================================

if __name__ == "__main__":
    from dataset import load_and_preprocess
    from search_engine import HybridSearchEngine

    print("=== EVALUASI SISTEM HYBRID TF-IDF + BERT ===")

    # Full dataset untuk evaluasi
    df = load_and_preprocess()
    engine = HybridSearchEngine(df)

    summary, details = evaluate_system(engine)

    print(f"\n--- HASIL EVALUASI (K={K}) ---")
    print(f"Avg P@{K}  : {summary['Avg_P@K']:.4f}")
    print(f"Avg R@{K}  : {summary['Avg_R@K']:.4f}")
    print(f"MRR        : {summary['MRR']:.4f}")

    print("\n--- HASIL PER KUERI ---")
    for d in details:
        print(f"Kueri: '{d['query']}'")
        print(f"  P@K: {d['P@K']:.4f} | R@K: {d['R@K']:.4f} | "
              f"RR: {d['RR']:.4f} | Hits: {d['Hits']}")

    # =========================================================
    # MLFLOW TRACKING FOR DAGSHUB
    # =========================================================
    import dagshub
    import mlflow

    print("\nMenghubungkan ke DagsHub untuk menyimpan hasil eksperimen...")
    try:
        import socket
        # Cek koneksi ke dagshub.com secara cepat (timeout 3 detik) untuk menghindari hang lama
        socket.create_connection(("dagshub.com", 80), timeout=3.0)
        dagshub.init(repo_owner='Tesyaf', repo_name='UASTKI', mlflow=True)
    except Exception as e:
        print(f"\n⚠️ Gagal menghubungkan ke DagsHub (atau offline/timeout): {e}")
        print("Menjalankan MLflow secara lokal...")
        mlflow.set_tracking_uri("file:./mlruns")

    with mlflow.start_run(run_name="Hybrid_TFIDF_BERT_Evaluation"):
        # Log Parameters
        mlflow.log_param("Algorithm", "Hybrid TF-IDF + Sentence-BERT")
        mlflow.log_param("BERT_Model", "all-MiniLM-L6-v2")
        mlflow.log_param("W_TFIDF", 0.35)
        mlflow.log_param("W_BERT", 0.65)
        mlflow.log_param("NLP_Library", "NLTK + sentence-transformers")
        mlflow.log_param("Stemmer", "PorterStemmer")
        mlflow.log_param("Dataset_Size", len(df))
        mlflow.log_param("Evaluation_K", K)

        # Log Metrics
        mlflow.log_metric("Avg_Precision_at_K", summary['Avg_P@K'])
        mlflow.log_metric("Avg_Recall_at_K", summary['Avg_R@K'])
        mlflow.log_metric("MRR", summary['MRR'])

        # Log per-query metrics
        for i, d in enumerate(details):
            mlflow.log_metric(f"Q{i+1}_PK", d['P@K'])
            mlflow.log_metric(f"Q{i+1}_RK", d['R@K'])
            mlflow.log_metric(f"Q{i+1}_RR", d['RR'])

        print("\n✅ Metrik evaluasi berhasil diunggah ke DagsHub MLflow!")
