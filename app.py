"""
app.py (Revisi Total - Bulletproof Version)
===========================================
Aplikasi Streamlit untuk prediksi Customer Churn menggunakan model terbaik
(Random Forest hasil hyperparameter tuning) yang sudah disimpan oleh
training_pipeline.py.

Fitur aplikasi:
- Memuat model, scaler, encoder, dan daftar fitur dari artifacts.pkl
- Form input manual untuk prediksi 1 pelanggan
- Upload CSV untuk prediksi batch banyak pelanggan sekaligus
- Visualisasi feature importance & metrik model yang aman dari dimensi kolom mismatch
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# Konfigurasi halaman
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Customer Churn",
    page_icon="📉",
    layout="wide",
)

ARTIFACTS_PATH = "artifacts.pkl"

ORDINAL_COLS = ["gender", "device_type"]
OHE_COLS = ["acquisition_channel", "subscription_type", "payment_method"]

# Informasi default & batas nilai fitur untuk komponen form input manual
NUMERIC_FEATURE_INFO = {
    "age": ("Usia pelanggan", 18, 90, 35, 1),
    "total_visits": ("Total kunjungan", 0, 500, 20, 1),
    "avg_session_time": ("Rata-rata waktu sesi (menit)", 0.0, 120.0, 5.0, 0.1),
    "pages_per_session": ("Rata-rata halaman per sesi", 0.0, 50.0, 3.0, 0.1),
    "email_open_rate": ("Persentase email dibuka (0-1)", 0.0, 1.0, 0.3, 0.01),
    "email_click_rate": ("Persentase klik email (0-1)", 0.0, 1.0, 0.1, 0.01),
    "total_spent": ("Total pengeluaran", 0.0, 50000.0, 200.0, 10.0),
    "avg_order_value": ("Rata-rata nilai transaksi", 0.0, 5000.0, 50.0, 5.0),
    "support_tickets": ("Jumlah tiket dukungan", 0, 50, 1, 1),
    "delivery_delay_days": ("Keterlambatan pengiriman (hari)", 0, 30, 1, 1),
    "satisfaction_score": ("Skor kepuasan (1-5)", 1.0, 5.0, 3.0, 0.1),
    "nps_score": ("Net Promoter Score (0-10)", 0, 10, 5, 1),
    "marketing_spend_per_user": ("Biaya marketing per user", 0.0, 1000.0, 10.0, 1.0),
    "lifetime_value": ("Customer Lifetime Value", 0.0, 50000.0, 500.0, 10.0),
    "last_3_month_purchase_freq": ("Frekuensi beli 3 bulan terakhir", 0, 100, 3, 1),
    "is_premium_user": None,
    "discount_used": None,
    "refund_requested": None,
}

BINARY_FEATURES = ["is_premium_user", "discount_used", "refund_requested"]


# -----------------------------------------------------------------------
# Load artefak model (di-cache supaya ringan)
# -----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(ARTIFACTS_PATH):
        return None, [ARTIFACTS_PATH]

    artifacts = joblib.load(ARTIFACTS_PATH)

    required_keys = [
        "model", "scaler", "ordinal_encoder", "onehot_encoder",
        "feature_columns", "top_features",
    ]
    missing_keys = [k for k in required_keys if k not in artifacts]
    if missing_keys:
        return None, missing_keys

    # Sinkronisasi otomatis dimensi fitur model asli
    model = artifacts["model"]
    if hasattr(model, "feature_names_in_"):
        artifacts["top_features"] = list(model.feature_names_in_)
    elif hasattr(model, "n_features_in_"):
        expected_count = model.n_features_in_
        if len(artifacts["top_features"]) != expected_count:
            artifacts["top_features"] = artifacts["top_features"][:expected_count]

    return artifacts, []


def preprocess_input(df_raw: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Fungsi Preprocessing yang aman dari bentrokan dimensi data (Anti-Duplicate Columns).
    Menerapkan transformasi encoder, scaler, dan seleksi kolom penentu secara presisi.
    """
    df = df_raw.copy()

    # Pastikan kolom kategorikal utama tersedia untuk menghindari KeyError
    for col in ORDINAL_COLS + OHE_COLS:
        if col not in df.columns:
            # Isi sementara dengan kategori pertama yang valid dari encoder
            if col in ORDINAL_COLS:
                idx = ORDINAL_COLS.index(col)
                df[col] = artifacts["ordinal_encoder"].categories_[idx][0]
            else:
                idx = OHE_COLS.index(col)
                df[col] = artifacts["onehot_encoder"].categories_[idx][0]

    # Imputasi nilai kosong untuk tipe data numerik
    numeric_input_cols = [c for c in df.columns if c not in ORDINAL_COLS + OHE_COLS]
    for col in numeric_input_cols:
        if df[col].isnull().any():
            fill_val = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(fill_val) else fill_val)

    # 1. Transformasi Ordinal Encoding
    df_encoded = df.copy()
    df_encoded[ORDINAL_COLS] = artifacts["ordinal_encoder"].transform(df[ORDINAL_COLS])

    # 2. Transformasi One-Hot Encoding
    ohe = artifacts["onehot_encoder"]
    ohe_arr = ohe.transform(df[OHE_COLS])
    ohe_cols_out = ohe.get_feature_names_out(OHE_COLS)
    ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols_out, index=df.index)

    # Singkirkan kolom kategorikal string asal sebelum digabungkan hasil OHE-nya
    df_encoded = df_encoded.drop(columns=OHE_COLS)

    # Mencegah duplikasi: Jika di df_encoded sudah terlanjur ada nama kolom tiruan OHE, bersihkan!
    dup_cols = [c for c in ohe_cols_out if c in df_encoded.columns]
    if dup_cols:
        df_encoded = df_encoded.drop(columns=dup_cols)

    df_final = pd.concat([df_encoded, ohe_df], axis=1)

    # 3. Penyusunan Struktur Fitur Mutlak (Garansi 36 Kolom)
    # Membuat wadah dataframe kosong berbasis susunan feature_columns training
    X_scaled_input = pd.DataFrame(0.0, columns=artifacts["feature_columns"], index=df.index)
    for col in artifacts["feature_columns"]:
        if col in df_final.columns:
            val = df_final[col]
            if isinstance(val, pd.DataFrame):  # jika masih ada duplikasi tak terduga, ambil kolom pertama
                val = val.iloc[:, 0]
            X_scaled_input[col] = val.fillna(0.0)

    # 4. Transformasi Scaling menggunakan StandardScaler (Pasti 36 Fitur)
    scaled_arr = artifacts["scaler"].transform(X_scaled_input)
    scaled_df = pd.DataFrame(scaled_arr, columns=artifacts["feature_columns"], index=df.index)

    # 5. Filter hanya mengambil fitur yang diminta oleh model Random Forest
    return scaled_df[artifacts["top_features"]]


def predict(df_raw: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    X_ready = preprocess_input(df_raw, artifacts)
    model = artifacts["model"]
    pred = model.predict(X_ready)
    proba = model.predict_proba(X_ready)[:, 1] if hasattr(model, "predict_proba") else None

    result = df_raw.copy()
    result["Prediksi"] = np.where(pred == 1, "Churn", "Tidak Churn")
    if proba is not None:
        result["Probabilitas Churn"] = np.round(proba, 4)
    return result


# -----------------------------------------------------------------------
# Antarmuka Pengguna (UI Streamlit)
# -----------------------------------------------------------------------
st.title("📉 Aplikasi Prediksi Customer Churn")
st.markdown(
    "Aplikasi ini memprediksi status keberlanjutan pelanggan menggunakan model **Random Forest** "
    "hasil optimasi hyperparameter tuning."
)

artifacts, missing_files = load_artifacts()

if artifacts is None:
    st.error(
        "❌ File `artifacts.pkl` tidak ditemukan atau strukturnya tidak lengkap.\n\n"
        "Pastikan file tersebut berada di satu direktori kerja dengan script `app.py`. Komponen hilang:\n\n"
        + "\n".join(f"- {m}" for m in missing_files)
    )
    st.stop()

tab1, tab2, tab3 = st.tabs(["🧍 Prediksi Manual", "📂 Prediksi Batch (CSV)", "📊 Info Model"])

# --- TAB 1: Prediksi Manual ---
with tab1:
    st.subheader("Form Isian Data Pelanggan Baru")
    st.caption("Input ini dirancang dinamis menyesuaikan fitur utama yang dipilih model.")

    top_features = artifacts["top_features"]
    input_data = {}

    # Pengaturan grid form agar rapi (3 Kolom)
    cols = st.columns(3)
    col_idx = 0

    for feat in top_features:
        # Jika fitur tersebut adalah turunan OHE (mengandung '_'), lewati karena diisi lewat selectbox bawah
        if any(feat.startswith(ohe_c + "_") for ohe_c in OHE_COLS) or feat in ORDINAL_COLS or feat in OHE_COLS:
            continue

        target_col = cols[col_idx % 3]
        col_idx += 1

        if feat in BINARY_FEATURES:
            with target_col:
                val = st.selectbox(feat, options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            input_data[feat] = val
        elif feat in NUMERIC_FEATURE_INFO and NUMERIC_FEATURE_INFO[feat] is not None:
            label, min_v, max_v, default_v, step_v = NUMERIC_FEATURE_INFO[feat]
            with target_col:
                val = st.number_input(
                    f"{feat}", min_value=float(min_v), max_value=float(max_v),
                    value=float(default_v), step=float(step_v), help=label
                )
            input_data[feat] = val
        else:
            with target_col:
                val = st.number_input(f"{feat}", value=0.0)
            input_data[feat] = val

    st.divider()
    st.markdown("**Atribut Kategori Utama Pelanggan:**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        gender = st.selectbox("Gender", artifacts["ordinal_encoder"].categories_[0])
    with c2:
        device_type = st.selectbox("Device Type", artifacts["ordinal_encoder"].categories_[1])
    with c3:
        acquisition_channel = st.selectbox("Acquisition Channel", artifacts["onehot_encoder"].categories_[0])
    with c4:
        subscription_type = st.selectbox("Subscription Type", artifacts["onehot_encoder"].categories_[1])
    with c5:
        payment_method = st.selectbox("Payment Method", artifacts["onehot_encoder"].categories_[2])

    full_input = {
        "gender": gender,
        "device_type": device_type,
        "acquisition_channel": acquisition_channel,
        "subscription_type": subscription_type,
        "payment_method": payment_method,
    }
    full_input.update(input_data)

    if st.button("🔍 Jalankan Prediksi Tunggal", type="primary"):
        df_input = pd.DataFrame([full_input])
        result = predict(df_input, artifacts)

        pred_label = result.loc[0, "Prediksi"]
        proba = result.loc[0, "Probabilitas Churn"] if "Probabilitas Churn" in result.columns else None

        if pred_label == "Churn":
            st.error(f"⚠️ Pelanggan ini **berpotensi CHURN (Berhenti Berlangganan)**" + (f" (Probabilitas: {proba:.1%})" if proba is not None else ""))
        else:
            st.success(f"✅ Pelanggan ini **diprediksi AKTIF (Tetap Berlangganan)**" + (f" (Probabilitas Churn: {proba:.1%})" if proba is not None else ""))

        with st.expander("Lihat matriks baris input raw"):
            st.dataframe(df_input)

# --- TAB 2: Prediksi Batch ---
with tab2:
    st.subheader("Proses Banyak Data Sekaligus via Upload File")
    st.markdown("Unggah file CSV pelanggan Anda dengan struktur standar dataset asal:")
    
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write(f"Data Masuk: **{df_batch.shape[0]} Baris** & **{df_batch.shape[1]} Kolom**")
            st.dataframe(df_batch.head(3))

            if st.button("🚀 Proses Prediksi Massal", type="primary"):
                with st.spinner("Menghitung perkiraan model..."):
                    result_batch = predict(df_batch, artifacts)

                st.success(f"Selesai! Evaluasi rampung untuk {len(result_batch)} record pelanggan.")

                churn_count = (result_batch["Prediksi"] == "Churn").sum()
                col_a, col_b = st.columns(2)
                col_a.metric("Total Potensi Churn", f"{churn_count} User", f"{churn_count/len(result_batch):.1%} ancaman")
                col_b.metric("Total Bertahan", f"{len(result_batch) - churn_count} User")

                st.dataframe(result_batch)

                csv_download = result_batch.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Unduh File Hasil Prediksi (.CSV)",
                    data=csv_download,
                    file_name="hasil_prediksi_churn_batch.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Gagal memproses file CSV: {e}")

# --- TAB 3: Detail Struktur & Performa Model ---
with tab3:
    st.subheader("Informasi Kinerja & Komponen Model")
    metrics = artifacts.get("metrics")

    if metrics:
        st.markdown(f"**Nama Arsitektur:** {metrics.get('model_name', 'Random Forest Classifier')}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy Score", f"{metrics['accuracy']:.4f}")
        c2.metric("Precision Score", f"{metrics['precision']:.4f}")
        c3.metric("Recall Score", f"{metrics['recall']:.4f}")
        c4.metric("F1-Score", f"{metrics['f1_score']:.4f}")

        st.markdown("**Parameter Terpilih (Best Params):**")
        st.json(metrics.get("best_params", {}))

        if "confusion_matrix" in metrics:
            st.markdown("**Confusion Matrix:**")
            cm = np.array(metrics["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.imshow(cm, cmap="Blues", alpha=0.8)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11, fontweight="bold")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Tidak Churn", "Churn"])
            ax.set_yticklabels(["Tidak Churn", "Churn"])
            ax.set_xlabel("Hasil Prediksi")
            ax.set_ylabel("Data Aktual")
            st.pyplot(fig)
    else:
        st.info("Catatan histori performa model di dalam artifacts kosong.")

    model = artifacts["model"]
    if hasattr(model, "feature_importances_"):
        st.markdown("**Tingkat Signifikansi Fitur (Feature Importance):**")
        imp_df = pd.DataFrame({
            "Fitur": artifacts["top_features"],
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.barh(imp_df["Fitur"], imp_df["Importance"], color="#34495E")
        ax2.invert_yaxis()
        ax2.set_xlabel("Skala Importance")
        st.pyplot(fig2)

st.divider()
st.caption("Dibuat untuk Ujian Akhir Semester — Bengkel Koding Data Science, Universitas Dian Nuswantoro.")