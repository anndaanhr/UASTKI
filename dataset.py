import pandas as pd
import re
import os
import ast
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Pastikan resource NLTK sudah didownload
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextPreprocessor:
    def __init__(self):
        # Inisialisasi Stemmer bahasa Inggris
        self.stemmer = PorterStemmer()
        
        # Inisialisasi Stopword Remover bahasa Inggris
        self.stopwords = set(stopwords.words('english'))
        
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
        
        # 5. Stemming (NLTK PorterStemmer)
        stemmed_tokens = [self.stemmer.stem(word) for word in tokens]
        
        # Return tokens akhir
        return stemmed_tokens

def load_and_preprocess(filepath, sample_size=None, cache_path="preprocessed_dataset.csv"):
    if os.path.exists(cache_path):
        print(f"Loading preprocessed dataset from {cache_path}...")
        df = pd.read_csv(cache_path)
        # Parse list of strings back from string representation
        df['tokens'] = df['tokens'].apply(ast.literal_eval)
        
        # FIX: Isi nilai NaN dengan string kosong agar tidak error saat di-convert ke JSON
        df['id'] = df['id'].fillna('')
        df['title'] = df['title'].fillna('')
        df['url'] = df['url'].fillna('')
        df['source'] = df['source'].fillna('Unknown')
        df['author'] = df['author'].fillna('Unknown')
        df['date'] = df['date'].fillna('Unknown')
        df['license'] = df['license'].fillna('Unknown')
        df['text'] = df['text'].fillna('')
        
        if sample_size:
            df = df.head(sample_size)
        return df

    print(f"Loading dataset from {filepath}...")
    # Baca format JSONL
    df = pd.read_json(filepath, lines=True)
    
    # Isi nilai NaN dengan string default
    df['id'] = df['id'].fillna('')
    df['title'] = df['title'].fillna('')
    df['url'] = df['url'].fillna('')
    df['source'] = df['source'].fillna('Unknown')
    df['author'] = df['author'].fillna('Unknown')
    df['date'] = df['date'].fillna('Unknown')
    df['license'] = df['license'].fillna('Unknown')
    df['text'] = df['text'].fillna('')
    
    if sample_size:
        df = df.head(sample_size)
        
    # Menggabungkan teks yang relevan untuk pencarian (title, source, author, text)
    df['search_content'] = df['title'] + " " + df['source'] + " " + df['author'] + " " + df['text']
    
    preprocessor = TextPreprocessor()
    
    print("Mulai preprocessing teks (Cleansing, Stopword Removal, Stemming)...")
    
    # Menerapkan preprocessing
    df['tokens'] = df['search_content'].apply(preprocessor.preprocess)
    
    print("Preprocessing selesai!")
    
    # Save to cache 
    print(f"Saving preprocessed dataset to {cache_path}...")
    df.to_csv(cache_path, index=False)
        
    return df

if __name__ == "__main__":
    # Test script dengan 5 data pertama agar cepat
    test_df = load_and_preprocess("corpus.jsonl", sample_size=5)
    
    for idx, row in test_df.iterrows():
        print(f"Title: {row['title']}")
        print(f"Tokens: {row['tokens'][:10]}...") # Hanya print 10 token pertama agar tidak spam
        print("-" * 50)

