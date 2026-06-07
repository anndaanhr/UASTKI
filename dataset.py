# -*- coding: utf-8 -*-
"""
Dataset Module — Middle East News Corpus
Download, cache, dan preprocessing dataset berita Timur Tengah.
"""

import requests
import json
import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from pathlib import Path

# Pastikan resource NLTK sudah didownload
for res in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f'corpora/{res}' if res == "stopwords" else f'tokenizers/{res}')
    except LookupError:
        nltk.download(res, quiet=True)

# =========================================================
# KONFIGURASI
# =========================================================
DATASET_URL = "https://raw.githubusercontent.com/Tesyaf/dataset_tki/refs/heads/main/corpus.jsonl"
CORPUS_PATH = "corpus.csv"
PREPROCESSED_PATH = "corpus_preprocessed.csv"
MIN_WORD_COUNT = 100

# =========================================================
# STEMMER & STOPWORDS
# =========================================================
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english")) | {
    "paper", "study", "research", "result", "results", "show", "shows",
    "propose", "proposed", "approach", "method", "methods", "based",
    "using", "used", "use", "also", "however", "therefore", "thus",
    "among", "within", "without", "well", "two", "three", "one",
    "first", "second", "third", "et", "al", "fig", "table", "section",
    "arxiv", "abstract", "introduction", "conclusion", "http", "www",
}


def clean_text(text):
    """
    Preprocessing teks: lowercasing, hapus URL/angka/simbol,
    tokenisasi, stopword removal, stemming.
    Digunakan untuk dokumen dan query.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) >= 2]
    tokens = [t for t in tokens if t not in stop_words]  # re-filter post-stemming
    return " ".join(tokens)


def download_dataset():
    """
    Download dataset berita Timur Tengah dari GitHub dan simpan ke corpus.csv.
    Hanya dokumen dengan word count >= MIN_WORD_COUNT yang disimpan.
    """
    if os.path.exists(CORPUS_PATH):
        print(f"[INFO] Memuat dataset dari cache: {CORPUS_PATH}")
        df = pd.read_csv(CORPUS_PATH)
        return df

    print("[INFO] Mengunduh dataset berita Timur Tengah ...")
    response = requests.get(DATASET_URL, timeout=120)
    response.raise_for_status()

    records = []
    for line in response.text.strip().split('\n'):
        try:
            item = json.loads(line)
            content = item.get("text", "")
            wc = len(content.split())
            if wc >= MIN_WORD_COUNT:
                records.append({
                    "doc_id": item.get("id"),
                    "title": item.get("title", "").strip(),
                    "url": item.get("url", ""),
                    "author": item.get("author", ""),
                    "date": item.get("date", ""),
                    "content": content,
                    "word_count": wc,
                    "source": item.get("source", "News"),
                    "license": item.get("license", ""),
                })
        except json.JSONDecodeError:
            continue

    df = pd.DataFrame(records)
    df.to_csv(CORPUS_PATH, index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] {len(df):,} dokumen disimpan ke {CORPUS_PATH}")
    return df


def load_and_preprocess(sample_size=None):
    """
    Load dataset dan lakukan preprocessing.
    Menggabungkan title (2x bobot) + content, lalu clean_text().
    Hasilnya di-cache ke corpus_preprocessed.csv.
    
    :param sample_size: Jika None, gunakan seluruh dataset (full).
    :return: DataFrame dengan kolom processed_text dan term_count.
    """
    if os.path.exists(PREPROCESSED_PATH):
        print(f"[INFO] Memuat data preprocessed dari cache: {PREPROCESSED_PATH}")
        df_proc = pd.read_csv(PREPROCESSED_PATH)

        # Fill NaN values
        for col in ['doc_id', 'title', 'url', 'author', 'date', 'content',
                     'source', 'license', 'processed_text']:
            if col in df_proc.columns:
                df_proc[col] = df_proc[col].fillna('')

        if sample_size:
            df_proc = df_proc.head(sample_size)
        return df_proc

    # Download dulu jika belum ada
    df = download_dataset()

    if sample_size:
        df_proc = df.head(sample_size).copy()
    else:
        df_proc = df.copy()

    print(f"[INFO] Preprocessing {len(df_proc):,} dokumen ...")

    # Gabungkan title (2x bobot) + content untuk indexing
    combined = df_proc['title'] + " " + df_proc['title'] + " " + df_proc['content']
    df_proc['processed_text'] = combined.apply(clean_text)
    df_proc['term_count'] = df_proc['processed_text'].apply(lambda x: len(x.split()))

    df_proc.to_csv(PREPROCESSED_PATH, index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] Data preprocessed disimpan ke {PREPROCESSED_PATH}")
    print(f"  Rata-rata term per dokumen: {df_proc['term_count'].mean():.0f}")

    return df_proc


if __name__ == "__main__":
    # Test: download + preprocess full dataset
    print("=== TEST DATASET MODULE ===")
    df = download_dataset()
    print(f"\nTotal dokumen  : {len(df):,}")
    print(f"Avg word count : {df['word_count'].mean():.1f}")
    print(f"Min word count : {df['word_count'].min()}")
    print(f"Max word count : {df['word_count'].max()}")
    print(f"Rentang tanggal: {df['date'].min()} s/d {df['date'].max()}")
    print(df[['doc_id', 'title', 'date', 'word_count']].head())

    print("\n=== PREPROCESSING ===")
    df_proc = load_and_preprocess()
    print(f"Rata-rata term per dokumen: {df_proc['term_count'].mean():.0f}")
    print(df_proc[['title', 'processed_text', 'term_count']].head(3))
