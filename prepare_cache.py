"""
Prepare Cache — Download dataset dan preprocess untuk mempercepat startup.
Jalankan sekali sebelum deploy untuk menghindari cold start yang lama.
"""
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from dataset import download_dataset, load_and_preprocess

print("Mempersiapkan cache seluruh dataset...")
print("Ini akan memakan waktu 1-3 menit (download + preprocessing)...")

# Step 1: Download dataset ke corpus.csv
df = download_dataset()
print(f"Dataset: {len(df):,} dokumen")

# Step 2: Preprocess full dataset (tanpa sampling)
df_proc = load_and_preprocess()
print(f"Preprocessed: {len(df_proc):,} dokumen")
print("Cache dataset berhasil dibuat dan disimpan!")
