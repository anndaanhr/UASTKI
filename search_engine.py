import pandas as pd
from rank_bm25 import BM25Okapi
from dataset import TextPreprocessor

class MovieSearchEngine:
    def __init__(self, df):
        """
        Inisialisasi Search Engine menggunakan model Okapi BM25.
        
        :param df: DataFrame pandas yang sudah memiliki kolom 'tokens' (hasil preprocessing).
        """
        self.df = df
        self.preprocessor = TextPreprocessor()
        
        # Menyiapkan corpus (kumpulan dokumen dalam bentuk list of tokens)
        print("Membangun Indeks BM25...")
        corpus = self.df['tokens'].tolist()
        
        # Inisialisasi model BM25
        self.bm25 = BM25Okapi(corpus)
        print("Indeks BM25 berhasil dibangun!")
        
    def search(self, query, top_k=10):
        """
        Melakukan pencarian dokumen berdasarkan kueri dengan skor BM25.
        
        :param query: String kueri pencarian.
        :param top_k: Jumlah hasil maksimal yang dikembalikan.
        :return: List of dictionaries berisi detail film beserta skor relevansinya.
        """
        # 1. Preprocessing kueri dengan cara yang persis sama dengan dokumen
        query_tokens = self.preprocessor.preprocess(query)
        
        # Jika kueri kosong setelah preprocessing
        if not query_tokens:
            return []
            
        # 2. Menghitung skor relevansi (BM25) untuk seluruh dokumen
        doc_scores = self.bm25.get_scores(query_tokens)
        
        # 3. Gabungkan skor dengan dataframe asli
        results_df = self.df.copy()
        results_df['relevance_score'] = doc_scores
        
        # 4. Filter dokumen yang memiliki relevansi > 0, lalu urutkan dari skor tertinggi
        top_results = results_df[results_df['relevance_score'] > 0].sort_values(
            by='relevance_score', ascending=False
        ).head(top_k)
        
        # Mengembalikan hasil dan metadata
        columns_to_return = ['id', 'title', 'url', 'source', 'author', 'date', 'license', 'text', 'relevance_score']
        
        # Ensure columns exist, if not fill with default to prevent KeyError
        for col in columns_to_return:
            if col not in top_results.columns:
                top_results[col] = "Unknown"
                
        results_list = top_results[columns_to_return].to_dict('records')
        
        return {
            "results": results_list,
            "query_tokens": query_tokens
        }

if __name__ == "__main__":
    from dataset import load_and_preprocess
    
    # Unit Test untuk memastikan BM25 bekerja dengan baik
    print("=== PENGUJIAN STEP 2: SEARCH ENGINE BM25 ===")
    
    # Gunakan sampel data untuk pengujian lokal agar lebih cepat
    df = load_and_preprocess("corpus.jsonl", sample_size=100) 
    
    engine = MovieSearchEngine(df)
    
    queries_to_test = [
        "semantic web policies",
        "electronic government in saudi arabia",
        "neural network architectures"
    ]
    
    for q in queries_to_test:
        print(f"\nKueri: '{q}'")
        results = engine.search(q, top_k=3)
        if not results:
            print("  -> Tidak ada dokumen yang relevan.")
        else:
            for rank, res in enumerate(results['results'], 1):
                print(f"  {rank}. {res['title']} (Score: {res['relevance_score']:.4f})")
                print(f"     Source: {res['source']}")
                print(f"     Text  : {res['text'][:100]}...")

