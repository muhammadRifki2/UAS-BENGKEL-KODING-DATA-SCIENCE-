# Deployment — Prediksi Customer Churn (Streamlit Cloud)

Paket ini berisi semua yang dibutuhkan untuk menyelesaikan **Poin Penugasan D.4 (Deployment)**
dari soal UAS Bengkel Koding Data Science.

## Isi Folder

```
deployment/
├── app.py                 # Aplikasi Streamlit (form manual + upload CSV batch)
├── artifacts.pkl           # SATU file berisi model + scaler + encoder + metrik
├── requirements.txt       # Dependensi untuk Streamlit Cloud
└── training_pipeline.py   # Script untuk training ulang & menghasilkan artifacts.pkl
```

Cuma 4 file. `artifacts.pkl` adalah gabungan dari model, scaler, encoder, daftar
fitur, dan metrik evaluasi — supaya tidak perlu mengelola banyak file `.pkl` terpisah.

## Status Model — SUDAH DI-GENERATE dengan Dataset Asli ✅

File `artifacts.pkl` di paket ini **sudah berisi model hasil training dengan dataset asli**
 (`Sales - Marketing customer dataset.csv`, 15.000 baris), bukan dummy lagi.
Hasil evaluasinya **identik** dengan yang ada di notebook:

| Metrik | Nilai |
|---|---|
| Model | Random Forest (Tuning) |
| Accuracy | 0.8674 |
| Precision | 0.5098 |
| Recall | 0.4286 |
| F1-Score | 0.4657 |
| Best params | `n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_depth=None` |

App `app.py` juga sudah ditest end-to-end dengan model ini (termasuk skenario data
yang mengandung missing value, seperti dataset asli) dan terbukti berjalan tanpa error.

**Tidak perlu jalankan ulang `training_pipeline.py`** kecuali  ingin retrain
dengan perubahan (misal dataset baru, atau ingin coba model lain). Script ini tetap
disertakan untuk dokumentasi/reproducibility — salah satu poin yang biasanya dinilai
dalam tugas deployment.

## Langkah 1 — Tes Lokal

```bash
streamlit run app.py
```

Buka browser ke `http://localhost:8501`. Pastikan:
- Tab **Prediksi Manual** bisa menghasilkan prediksi tanpa error.
- Tab **Prediksi Batch (CSV)** bisa menerima file CSV dan menghasilkan output.
- Tab **Info Model** menampilkan metrik dan feature importance.

## Langkah 2 — Upload ke GitHub

1. Buat repository baru di GitHub (boleh publik).
2. Upload isi folder `deployment/` ini **beserta `artifacts.pkl`** (jangan dimasukkan ke `.gitignore`, ukurannya ~16 MB, masih jauh di bawah limit 100 MB GitHub).
3. Struktur di GitHub harus seperti ini di root repo:
   ```
   app.py
   artifacts.pkl
   requirements.txt
   training_pipeline.py
   ```

## Langkah 3 — Deploy ke Streamlit Cloud

1. Buka [streamlit.io/cloud](https://streamlit.io/cloud) → login dengan akun GitHub.
2. Klik **New app**.
3. Pilih repository, branch (`main`), dan **Main file path**: `app.py`.
4. Klik **Deploy**.
5. Tunggu proses build (install dependencies dari `requirements.txt`).
6. Setelah selesai, aplikasi akan punya URL publik seperti:
   `https://nama-app-rifki.streamlit.app`

## Troubleshooting Umum

| Masalah | Solusi |
|---|---|
| `ModuleNotFoundError` saat deploy | Pastikan semua library ada di `requirements.txt` |
| Model gagal load (`FileNotFoundError`) | Pastikan `artifacts.pkl` ikut di-push ke GitHub, satu folder dengan `app.py` |
| Error versi scikit-learn saat load `.pkl` | Pastikan versi scikit-learn di `requirements.txt` sama dengan yang dipakai saat training |
| App "lambat" pertama kali dibuka | Normal — Streamlit Cloud "tidur" jika tidak diakses lama, butuh beberapa detik untuk bangun |

## Catatan Teknis Tambahan

- Form prediksi manual hanya menampilkan **15 fitur hasil feature selection**
  (yang paling berpengaruh ke prediksi churn), sesuai pipeline notebook.
- Field tambahan (gender, device_type, acquisition_channel, dll) tetap diminta
  di form karena dibutuhkan untuk proses encoding, meskipun tidak masuk top 15 fitur.
- Mode "Prediksi Batch CSV" berguna untuk demo ke dosen — bisa upload sebagian
  data test (`X_test` asli, tanpa kolom churn) dan tunjukkan hasilnya langsung.
- **Penanganan missing value saat prediksi**: dataset asli memang punya beberapa
  baris dengan nilai kosong (terlihat di hasil EDA). `app.py` sudah dilengkapi
  imputasi otomatis (median untuk numerik, kategori pertama untuk kategorikal)
  saat menerima CSV mentah, jadi tidak perlu membersihkan data sebelum upload.
  Ini sudah ditest langsung dengan dataset asli 15.000 baris (3.690 nilai kosong)
  dan berjalan tanpa error.
