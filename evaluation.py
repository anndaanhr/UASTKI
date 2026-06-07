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
    "film horor seram tentang hantu": [
        "Mangkujiwo", "May the Devil Take You: Chapter Two", 
        "Rasuk 2", "Surat dari Kematian"
    ],
    "cerita cinta SMA yang romantis": [
        "Dilan 1991", "Dilan 1990", "Mariposa", "Milea"
    ],
    "film komedi lucu bikin ngakak": [
        "Guru-Guru Gokil", "Toko Barang Mantan", "Kapal Goyang Kapten"
    ],
    "petualangan anak muda": [
        "Anak Garuda", "Titus: Mystery of the Enygma"
    ],
    "cerita sedih keluarga": [
        "Nanti Kita Cerita Tentang Hari Ini", "Keluarga Cemara", "Si Doel the Movie"
    ],
    "kisah pahlawan indonesia": [
        "Gundala", "Sultan Agung: Tahta, Perjuangan, Cinta", "Wiro Sableng 212"
    ],
    "misteri pembunuhan": [
        "4 Mantan", "Pintu Merah"
    ],
    "film dokumenter biografi": [
        "Habibie & Ainun 3", "Susi Susanti: Love All", "A Man Called Ahok"
    ],
    "aksi perampokan dan laga": [
        "Hit & Run", "Darah Daging", "The Night Comes for Us"
    ],
    "cinta beda agama": [
        "Ajari Aku Islam"
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
    df = load_and_preprocess("indonesian_movies.csv", sample_size=500) 
    engine = MovieSearchEngine(df)
    
    summary, details = evaluate_system(engine)
    
    print("\n--- HASIL EVALUASI KESELURUHAN ---")
    print(f"MAP              : {summary['MAP']:.4f}")
    print(f"Avg Precision@5  : {summary['Avg_Precision@5']:.4f}")
    print(f"Avg Precision@10 : {summary['Avg_Precision@10']:.4f}")
    print(f"Avg NDCG@10      : {summary['Avg_NDCG@10']:.4f}")
    
    print("\n--- HASIL PER KUERI ---")
    for d in details:
        print(f"Kueri: '{d['query']}'")
        print(f"  P@5: {d['p5']:.2f} | P@10: {d['p10']:.2f} | AP: {d['ap']:.2f} | NDCG: {d['ndcg']:.2f}")
