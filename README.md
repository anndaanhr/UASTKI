<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/1200px-Google_2015_logo.svg.png" alt="Google Logo" width="200"/>
  <h1>Google Search 2025: Academic Engine</h1>
  <p><i>Sistem Temu Kembali Informasi (STKI) Cerdas untuk Jurnal Akademik dengan Antarmuka Futuristik</i></p>
  
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com/)
  [![NLTK](https://img.shields.io/badge/NLTK-NLP-yellow.svg)](https://www.nltk.org/)
  [![BM25](https://img.shields.io/badge/Algorithm-Okapi_BM25-red.svg)](https://en.wikipedia.org/wiki/Okapi_BM25)
  [![DagsHub](https://img.shields.io/badge/MLOps-DagsHub-purple.svg)](https://dagshub.com/)
</div>

<br/>

## 🌟 Tentang Proyek
Proyek ini adalah implementasi **Sistem Temu Kembali Informasi (STKI)** khusus untuk pencarian literatur akademik dan *research papers*. Dibangun sebagai pemenuhan Tugas Akhir/UAS, sistem ini memadukan ketangguhan algoritma pencarian probabilistik dengan keindahan antarmuka web modern yang terinspirasi dari desain konsep **Google Search Engine 2025**.

## ✨ Fitur Utama
- **🔬 Algoritma Okapi BM25**: Menggunakan metode perankingan probabilistik standar industri untuk akurasi pencarian tinggi.
- **🧠 NLTK Preprocessing Pipeline**: Pemrosesan teks bahasa Inggris menyeluruh (*Case Folding, Cleansing, Tokenization, Stopword Removal, dan PorterStemmer*).
- **🎨 UI/UX Google 2025**: Antarmuka bersih bergaya Material Design 3, mendukung *sticky search bar*, daftar hasil vertikal yang rapi, dan *AI Overview* simulasi.
- **📖 Knowledge Panel Interaktif**: Menampilkan detail metadata jurnal (Abstrak, Penulis, URL, dan Lisensi) di panel sisi kanan layaknya fitur profil Google.
- **🚀 Ultra-Fast Caching**: Data otomatis di-*cache* ke bentuk CSV (tokenized) pada awal *boot*, membuat pencarian berikutnya secepat kilat.
- **📊 Evaluasi DagsHub & MLflow**: Kode pengujian siap pakai untuk menghitung metrik **MAP, Precision@5, Precision@10, dan NDCG@10**, terintegrasi langsung dengan DagsHub.

## 🛠️ Arsitektur Teknologi
*   **Backend**: Python, FastAPI, Uvicorn
*   **NLP & IR**: NLTK, rank-bm25, Pandas
*   **Frontend**: HTML5, Vanilla CSS3 (Material 3 styling), Vanilla JavaScript
*   **MLOps**: DagsHub, MLflow

## 📂 Struktur Repositori
```text
📦 UASTKI
 ┣ 📂 static/               # Aset Statis (CSS, JS)
 ┣ 📂 templates/            # File HTML (Jinja2)
 ┣ 📜 app.py                # File utama FastAPI Web Server
 ┣ 📜 dataset.py            # Modul NLP Preprocessing & Data Loading
 ┣ 📜 search_engine.py      # Implementasi algoritma Okapi BM25
 ┣ 📜 evaluation.py         # Skrip pengujian & integrasi MLflow/DagsHub
 ┣ 📜 corpus.jsonl          # Dataset asli (8000+ dokumen akademik)
 ┣ 📜 preprocessed_dataset.csv # Dataset bersih (berisi token term)
 ┗ 📜 requirements.txt      # Dependensi Python
```

## 🚀 Panduan Instalasi (Lokal)
1. Kloning repositori ini:
   ```bash
   git clone https://github.com/anndaanhr/UASTKI.git
   cd UASTKI
   ```
2. Instal pustaka yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan server lokal:
   ```bash
   uvicorn app:app --reload
   ```
   *(Catatan: Saat dijalankan pertama kali, sistem mungkin butuh waktu 10-30 detik untuk mengunduh pustaka NLTK dan membuat file cache CSV).*
4. Buka browser dan akses: `http://localhost:8000`

## 📈 Evaluasi Model
Kami telah merancang 10 *Ground Truth Queries* yang menguji kehandalan sistem. Untuk melihat hasil evaluasinya, jalankan perintah:
```bash
python evaluation.py
```
Skrip ini akan menghitung skor *Mean Average Precision* (MAP) dan metrik lainnya, lalu akan mencatat riwayat eksperimen tersebut ke MLflow DagsHub secara otomatis.

---
<div align="center">
  <b>Dibangun dengan ❤️ untuk Evaluasi Ujian Akhir Semester STKI</b>
</div>
