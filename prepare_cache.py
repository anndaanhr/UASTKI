from dataset import load_and_preprocess

print("Mempersiapkan cache seluruh dataset. Ini akan memakan waktu 1-3 menit...")
# Panggil tanpa sample_size agar Sastrawi memproses semua 1200+ baris
# dan menyimpan hasilnya ke preprocessed_dataset.csv
df = load_and_preprocess("indonesian_movies.csv")
print("Cache dataset berhasil dibuat dan disimpan!")
