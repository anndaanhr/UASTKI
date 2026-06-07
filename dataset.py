import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class TextPreprocessor:
    def __init__(self):
        # Inisialisasi Stemmer
        stemmer_factory = StemmerFactory()
        self.stemmer = stemmer_factory.create_stemmer()
        
        # Inisialisasi Stopword Remover
        stopword_factory = StopWordRemoverFactory()
        # Bisa juga menambahkan stopword kustom jika perlu
        self.stopwords = stopword_factory.get_stop_words()
        
    def preprocess(self, text):
        if not isinstance(text, str):
            text = str(text)
            
        # 1. Case Folding
        text = text.lower()
        
        # 2. Cleansing (Hapus tanda baca, angka, dan karakter aneh)
        # Hanya menyisakan huruf a-z dan spasi
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Hapus spasi berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 3. Stopword Removal & 4. Tokenization
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stopwords]
        
        # Gabungkan kembali untuk proses stemming
        text = ' '.join(tokens)
        
        # 5. Stemming (Sastrawi)
        text = self.stemmer.stem(text)
        
        # Return tokens akhir
        return text.split()

import os
import ast

def load_and_preprocess(filepath, sample_size=None, cache_path="preprocessed_dataset.csv"):
    if os.path.exists(cache_path):
        print(f"Loading preprocessed dataset from {cache_path}...")
        df = pd.read_csv(cache_path)
        # Parse list of strings back from string representation
        df['tokens'] = df['tokens'].apply(ast.literal_eval)
        
        # FIX: Isi nilai NaN dengan string kosong agar tidak error saat di-convert ke JSON
        df['title'] = df['title'].fillna('')
        df['genre'] = df['genre'].fillna('')
        df['description'] = df['description'].fillna('')
        df['directors'] = df['directors'].fillna('Unknown')
        df['actors'] = df['actors'].fillna('Unknown')
        df['year'] = df['year'].fillna('Unknown')
        df['rating'] = df['rating'].fillna('Unrated')
        
        if sample_size:
            df = df.head(sample_size)
        return df

    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Isi nilai NaN dengan string kosong
    df['title'] = df['title'].fillna('')
    df['genre'] = df['genre'].fillna('')
    df['description'] = df['description'].fillna('')
    
    if sample_size:
        df = df.head(sample_size)
        
    # Menggabungkan teks yang relevan untuk pencarian
    df['search_content'] = df['title'] + " " + df['genre'] + " " + df['description']
    
    preprocessor = TextPreprocessor()
    
    print("Mulai preprocessing teks (Cleansing, Stopword Removal, Stemming)...")
    print("Mohon tunggu, proses Stemming Sastrawi membutuhkan waktu.")
    
    # Menerapkan preprocessing
    df['tokens'] = df['search_content'].apply(preprocessor.preprocess)
    
    print("Preprocessing selesai!")
    
    if not sample_size:
        # Save to cache if processing full dataset
        print(f"Saving preprocessed dataset to {cache_path}...")
        df.to_csv(cache_path, index=False)
        
    return df

if __name__ == "__main__":
    # Test script dengan 5 data pertama agar cepat
    test_df = load_and_preprocess("indonesian_movies.csv", sample_size=5)
    
    for idx, row in test_df.iterrows():
        print(f"Title: {row['title']}")
        print(f"Tokens: {row['tokens']}")
        print("-" * 50)
