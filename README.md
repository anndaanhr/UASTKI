# 📚 Research Archive (Information Retrieval System)

[![DagsHub Data](https://dagshub.com/anndaanhr/UASTKI/badges/data)](https://dagshub.com/anndaanhr/UASTKI)
[![MLflow Experiments](https://dagshub.com/anndaanhr/UASTKI/badges/experiments)](https://dagshub.com/anndaanhr/UASTKI/experiments)

Sistem Temu Kembali Informasi (Search Engine) yang menggunakan algoritma **Okapi BM25** dan pustaka **NLTK** untuk memproses serta mencari data jurnal/artikel penelitian bahasa Inggris. Proyek ini dibuat untuk Ujian Akhir Semester Temu Kembali Informasi (UASTKI).

## 🚀 Fitur Utama
* **Algoritma BM25**: Pencarian dengan mempertimbangkan frekuensi kata dan panjang dokumen.
* **NLTK Pipeline**: Menggunakan `PorterStemmer` dan penghapusan *stopwords* bahasa Inggris untuk memproses teks abstrak dan judul.
* **Glassmorphism UI**: Antarmuka modern yang interaktif dan responsif, dibangun dengan CSS Vanilla tanpa framework tambahan.
* **MLflow Tracking**: Evaluasi skor (MAP, NDCG, Precision) otomatis tercatat secara live menggunakan DagsHub MLflow.

## ⚙️ Cara Menjalankan Server
Pastikan semua library terinstall:
```bash
pip install fastapi uvicorn pandas nltk rank_bm25 dagshub mlflow
```

Jalankan server menggunakan uvicorn:
```bash
uvicorn app:app --reload
```
Akses di browser melalui `http://localhost:8000`.

## 📊 Evaluasi Eksperimen (DagsHub MLflow)
Untuk menjalankan evaluasi dan mengirim metriknya secara otomatis ke DagsHub:
```bash
python evaluation.py
```
*Catatan: Saat dijalankan pertama kali, terminal mungkin akan membuka browser untuk otentikasi DagsHub Anda.*

---
*Ditenagai oleh [DagsHub](https://dagshub.com/)*
