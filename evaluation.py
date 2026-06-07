import numpy as np

# =========================================================
# FUNGSI METRIK EVALUASI
# =========================================================

def precision_at_k(relevant_docs, retrieved_docs, k):
    """
    Menghitung Precision@K
    :param relevant_docs: Set judul dokumen yang benar-benar relevan
    :param retrieved_docs: List judul dokumen yang dikembalikan sistem
    :param k: Nilai K untuk batas jumlah dokumen yang dievaluasi
    """
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = [doc for doc in retrieved_k if doc in relevant_docs]
    return len(relevant_retrieved) / k if k > 0 else 0.0

def average_precision(relevant_docs, retrieved_docs):
    """
    Menghitung Average Precision (AP) untuk satu kueri
    """
    hits = 0
    sum_precs = 0
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            hits += 1
            sum_precs += hits / (i + 1.0)
    if not relevant_docs:
        return 0.0
    return sum_precs / len(relevant_docs)

def mean_average_precision(queries_ap):
    """
    Menghitung Mean Average Precision (MAP) untuk seluruh kueri
    """
    if not queries_ap: return 0.0
    return sum(queries_ap) / len(queries_ap)

def dcg_at_k(relevant_docs, retrieved_docs, k):
    """
    Menghitung Discounted Cumulative Gain (DCG) pada peringkat K.
    """
    retrieved_k = retrieved_docs[:k]
    dcg = 0.0
    for i, doc in enumerate(retrieved_k):
        if doc in relevant_docs:
            rel = 1.0 # Relevansi binary (1 relevan, 0 tidak)
            dcg += rel / np.log2(i + 2) # index dimulai dari 0 -> log2(2), log2(3), dst.
    return dcg

def ndcg_at_k(relevant_docs, retrieved_docs, k):
    """
    Menghitung Normalized Discounted Cumulative Gain (NDCG) pada peringkat K.
    """
    dcg_max = 0.0
    # Hitung Ideal DCG (jika semua dokumen yang relevan ada di urutan teratas)
    for i in range(min(len(relevant_docs), k)):
        dcg_max += 1.0 / np.log2(i + 2)
        
    if not dcg_max:
        return 0.0
        
    return dcg_at_k(relevant_docs, retrieved_docs, k) / dcg_max


# =========================================================
# DUMMY GROUND TRUTH
# 10 Kueri dengan konteks pencarian yang berbeda
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

def evaluate_system(engine):
    """
    Fungsi utama untuk mengevaluasi Search Engine (BM25) 
    terhadap Ground Truth yang disiapkan.
    """
    metrics = {
        'Precision@5': [],
        'Precision@10': [],
        'AP': [],
        'NDCG@10': []
    }
    
    detailed_results = []
    
    for query, relevant_docs in GROUND_TRUTH.items():
        # Lakukan pencarian, ambil top 10
        search_data = engine.search(query, top_k=10)
        results = search_data['results']
        retrieved_docs = [res['title'] for res in results]
        
        # Hitung metrik
        p5 = precision_at_k(relevant_docs, retrieved_docs, 5)
        p10 = precision_at_k(relevant_docs, retrieved_docs, 10)
        ap = average_precision(relevant_docs, retrieved_docs)
        ndcg = ndcg_at_k(relevant_docs, retrieved_docs, 10)
        
        metrics['Precision@5'].append(p5)
        metrics['Precision@10'].append(p10)
        metrics['AP'].append(ap)
        metrics['NDCG@10'].append(ndcg)
        
        detailed_results.append({
            'query': query,
            'p5': p5,
            'p10': p10,
            'ap': ap,
            'ndcg': ndcg
        })
        
    # Hitung rata-rata
    summary = {
        'MAP': float(mean_average_precision(metrics['AP'])),
        'Avg_Precision@5': float(np.mean(metrics['Precision@5'])),
        'Avg_Precision@10': float(np.mean(metrics['Precision@10'])),
        'Avg_NDCG@10': float(np.mean(metrics['NDCG@10']))
    }
    
    return summary, detailed_results

if __name__ == "__main__":
    from dataset import load_and_preprocess
    from search_engine import MovieSearchEngine
    
    print("=== PENGUJIAN STEP 3: SISTEM EVALUASI ===")
    
    # Supaya cepat untuk testing, sampel 500 saja. Untuk real evaluation, gunakan seluruh dataset.
    df = load_and_preprocess("corpus.jsonl", sample_size=500) 
    engine = MovieSearchEngine(df)
    
    summary, details = evaluate_system(engine)
    
    print("\n--- HASIL EVALUASI KESELURUHAN ---")
    print(f"MAP              : {summary['MAP']:.4f}")
    print(f"Avg Precision@5  : {summary['Avg_Precision@5']:.4f}")
    print(f"Avg Precision@10 : {summary['Avg_Precision@10']:.4f}")
    print(f"Avg NDCG@10      : {summary['Avg_NDCG@10']:.4f}")

    # =========================================================
    # MLFLOW TRACKING FOR DAGSHUB
    # =========================================================
    import dagshub
    import mlflow

    print("\nMenghubungkan ke DagsHub untuk menyimpan hasil eksperimen...")
    # Initialize DagsHub tracking (this may open a browser window for authentication)
    dagshub.init(repo_owner='anndaanhr', repo_name='UASTKI', mlflow=True)

    with mlflow.start_run(run_name="BM25_Evaluation_Research_Papers"):
        # Log Parameters
        mlflow.log_param("Algorithm", "Okapi BM25")
        mlflow.log_param("NLP_Library", "NLTK")
        mlflow.log_param("Stemmer", "PorterStemmer")
        mlflow.log_param("Dataset_Size", len(df))
        
        # Log Metrics
        mlflow.log_metric("MAP", summary['MAP'])
        mlflow.log_metric("Avg_Precision_at_5", summary['Avg_Precision@5'])
        mlflow.log_metric("Avg_Precision_at_10", summary['Avg_Precision@10'])
        mlflow.log_metric("Avg_NDCG_at_10", summary['Avg_NDCG@10'])
        
        print("\n✅ Metrik evaluasi berhasil diunggah ke DagsHub MLflow!")

    print("\n--- HASIL PER KUERI ---")
    for d in details:
        print(f"Kueri: '{d['query']}'")
        print(f"  P@5: {d['p5']:.2f} | P@10: {d['p10']:.2f} | AP: {d['ap']:.2f} | NDCG: {d['ndcg']:.2f}")
