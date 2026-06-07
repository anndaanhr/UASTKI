<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/1200px-Google_2015_logo.svg.png" alt="Google Logo" width="200"/>
  <h1>Google Search 2025: Academic Engine</h1>
  <p><i>Sistem Temu Kembali Informasi (STKI) Cerdas untuk Jurnal Akademik dengan Antarmuka Futuristik — Hybrid TF-IDF + Sentence-BERT</i></p>
  
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com/)
  [![NLTK](https://img.shields.io/badge/NLTK-NLP-yellow.svg)](https://www.nltk.org/)
  [![Hybrid](https://img.shields.io/badge/Algorithm-Hybrid_TF--IDF_%2B_BERT-red.svg)](#)
  [![DagsHub](https://img.shields.io/badge/MLOps-DagsHub-purple.svg)](https://dagshub.com/)
</div>

<br/>

## 🌟 Tentang Proyek
Proyek ini adalah implementasi **Sistem Temu Kembali Informasi (STKI)** khusus untuk pencarian literatur akademik dan *research papers*. Dibangun sebagai pemenuhan Tugas Akhir/UAS, sistem ini memadukan algoritma pencarian **Hybrid TF-IDF + Sentence-BERT** (`all-MiniLM-L6-v2`) dengan keindahan antarmuka web modern yang terinspirasi dari desain konsep **Google Search Engine 2025**.

## ✨ Fitur Utama
- **🔬 Hybrid TF-IDF + Sentence-BERT**: Menggabungkan TF-IDF (bobot 0.35) dan Sentence-BERT (bobot 0.65) untuk pencarian yang akurat secara leksikal dan semantik.
- **🧠 NLTK Preprocessing Pipeline**: Pemrosesan teks bahasa Inggris menyeluruh (*Case Folding, Cleansing, Tokenization, Stopword Removal, dan PorterStemmer*).
- **🎨 UI/UX Google 2025**: Antarmuka bersih bergaya Material Design 3, mendukung *sticky search bar*, daftar hasil vertikal yang rapi, *AI Overview* simulasi, dan *Knowledge Panel* interaktif.
- **📊 Score Breakdown**: Menampilkan detail skor TF-IDF dan BERT per hasil pencarian.
- **🚀 Ultra-Fast Caching**: Data otomatis di-*cache* ke bentuk CSV (tokenized) pada awal *boot*, membuat pencarian berikutnya secepat kilat.
- **📊 Evaluasi DagsHub & MLflow**: Kode pengujian siap pakai untuk menghitung metrik **P@K, R@K, dan MRR**, terintegrasi langsung dengan DagsHub.

## 🛠️ Arsitektur Teknologi
*   **Backend**: Python, FastAPI, Uvicorn
*   **NLP & IR**: NLTK, scikit-learn (TF-IDF), sentence-transformers (BERT), Pandas
*   **Frontend**: HTML5, Vanilla CSS3 (Material 3 styling), Vanilla JavaScript
*   **MLOps**: DagsHub, MLflow

## 📂 Struktur Repositori
```text
📦 UASTKI
 ┣ 📂 static/               # Aset Statis (CSS, JS)
 ┣ 📂 templates/            # File HTML (Jinja2)
 ┣ 📜 app.py                # File utama FastAPI Web Server
 ┣ 📜 dataset.py            # Modul NLP Preprocessing & Data Loading
 ┣ 📜 search_engine.py      # Implementasi algoritma Hybrid TF-IDF + BERT
 ┣ 📜 evaluation.py         # Skrip pengujian & integrasi MLflow/DagsHub
 ┣ 📜 prepare_cache.py      # Skrip persiapan cache dataset
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
3. (Opsional) Siapkan cache dataset terlebih dahulu agar startup lebih cepat:
   ```bash
   python prepare_cache.py
   ```
4. Jalankan server lokal:
   ```bash
   uvicorn app:app --reload
   ```
   *(Catatan: Saat dijalankan pertama kali, sistem mungkin butuh waktu 30-60 detik untuk mengunduh model BERT dan membuat embeddings).*
5. Buka browser dan akses: `http://localhost:8000`

## 📈 Evaluasi Model
Kami telah merancang 5 *Ground Truth Queries* yang menguji kehandalan sistem pada konteks berita Timur Tengah. Untuk melihat hasil evaluasinya, jalankan perintah:
```bash
python evaluation.py
```
Skrip ini akan menghitung skor *P@K, R@K, dan MRR*, lalu akan mencatat riwayat eksperimen tersebut ke MLflow DagsHub secara otomatis.

---
<div align="center">
  <b>Dibangun dengan ❤️ untuk Evaluasi Ujian Akhir Semester STKI</b>
</div>
