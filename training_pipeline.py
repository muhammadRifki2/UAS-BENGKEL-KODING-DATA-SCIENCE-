"""
training_pipeline.py
=====================
Script ini me-reproduksi seluruh pipeline preprocessing + training + hyperparameter
tuning dari notebook UAS_BengkelKoding.ipynb, lalu menyimpan model TERBAIK
(Random Forest setelah tuning) beserta seluruh objek pendukung (scaler, encoder,
daftar fitur) ke dalam SATU file "artifacts.pkl" agar mudah dimuat oleh aplikasi
Streamlit (app.py).

Cara pakai:
    1. Pastikan file "Sales - Marketing customer dataset.csv" ada di folder yang sama
       dengan script ini.
    2. Jalankan:  python training_pipeline.py
    3. Output: "artifacts.pkl" — berisi dictionary dengan key:
         - model             -> model Random Forest (tuning) terbaik
         - scaler            -> StandardScaler yang sudah di-fit
         - ordinal_encoder   -> OrdinalEncoder (gender, device_type)
         - onehot_encoder    -> OneHotEncoder (acquisition_channel, dll)
         - feature_columns   -> daftar kolom hasil encoding (urutan penting!)
         - top_features      -> daftar 15 fitur hasil feature selection
         - metrics           -> metrik evaluasi model terbaik (utk ditampilkan di app)
    4. Letakkan "artifacts.pkl" satu folder dengan "app.py" sebelum deploy.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = "Sales - Marketing customer dataset.csv"

DROP_COLS = ["customer_id", "signup_date", "last_purchase_date", "coupon_code", "country", "city"]
ORDINAL_COLS = ["gender", "device_type"]
OHE_COLS = ["acquisition_channel", "subscription_type", "payment_method"]

OUTLIER_COLS = [
    "age", "total_visits", "avg_session_time", "pages_per_session",
    "email_open_rate", "email_click_rate", "total_spent",
    "avg_order_value", "support_tickets", "delivery_delay_days",
    "satisfaction_score", "nps_score", "marketing_spend_per_user",
    "lifetime_value", "last_3_month_purchase_freq",
]


def load_and_clean_data(path: str) -> pd.DataFrame:
    print(f"[1/7] Memuat dataset dari '{path}' ...")
    df = pd.read_csv(path)
    print(f"      Shape awal: {df.shape}")

    # Handling missing value: numerik -> median, kategorikal -> mode
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include="object").columns

    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Handling duplikasi
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"      Duplikasi dihapus: {before - df.shape[0]} baris")

    # Handling outlier dengan IQR
    for col in OUTLIER_COLS:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print(f"      Shape setelah cleaning (missing+dup+outlier): {df.shape}")
    return df


def main():
    # 1. Load & cleaning
    # ---------------------------------------------------------------
    df = load_and_clean_data(DATA_PATH)

    # Drop kolom tidak relevan
    df_clean = df.drop(columns=DROP_COLS)

    # ---------------------------------------------------------------
    # 2. Tetapkan X, y dan train-test split (proporsi sama seperti notebook)
    # ---------------------------------------------------------------
    print("[2/7] Train-test split ...")
    X = df_clean.drop(columns=["churn"])
    y = df_clean["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Data latih: {X_train.shape[0]} | Data uji: {X_test.shape[0]}")

    # ---------------------------------------------------------------
    # 3. Encoding (setelah split, sesuai notebook)
    # ---------------------------------------------------------------
    print("[3/7] Encoding fitur kategorikal ...")
    oe = OrdinalEncoder()
    X_train[ORDINAL_COLS] = oe.fit_transform(X_train[ORDINAL_COLS])
    X_test[ORDINAL_COLS] = oe.transform(X_test[ORDINAL_COLS])

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train_ohe = ohe.fit_transform(X_train[OHE_COLS])
    X_test_ohe = ohe.transform(X_test[OHE_COLS])

    X_train_ohe_df = pd.DataFrame(
        X_train_ohe, columns=ohe.get_feature_names_out(OHE_COLS), index=X_train.index
    )
    X_test_ohe_df = pd.DataFrame(
        X_test_ohe, columns=ohe.get_feature_names_out(OHE_COLS), index=X_test.index
    )

    X_train = pd.concat([X_train.drop(columns=OHE_COLS), X_train_ohe_df], axis=1)
    X_test = pd.concat([X_test.drop(columns=OHE_COLS), X_test_ohe_df], axis=1)

    feature_columns = X_train.columns.tolist()
    print(f"      Total kolom setelah encoding: {len(feature_columns)}")

    # ---------------------------------------------------------------
    # 4. Scaling (setelah split, sesuai notebook)
    # ---------------------------------------------------------------
    print("[4/7] Scaling fitur (StandardScaler) ...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # ---------------------------------------------------------------
    # 5. Feature importance (Random Forest) -> top 15 fitur
    # ---------------------------------------------------------------
    print("[5/7] Menghitung feature importance & memilih top 15 fitur ...")
    rf_for_importance = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_for_importance.fit(X_train_scaled, y_train)

    feat_imp_df = pd.DataFrame({
        "Fitur": X_train.columns,
        "Importance": rf_for_importance.feature_importances_,
    }).sort_values(by="Importance", ascending=False)

    top_features = feat_imp_df.head(15)["Fitur"].tolist()
    print(f"      Top 15 fitur: {top_features}")

    X_train_selected = X_train_scaled[top_features]
    X_test_selected = X_test_scaled[top_features]

    # ---------------------------------------------------------------
    # 6. Hyperparameter tuning Random Forest (model terbaik di notebook)
    # ---------------------------------------------------------------
    print("[6/7] Hyperparameter tuning Random Forest (RandomizedSearchCV) ...")
    param_grid_rf = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    rs_rf = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=param_grid_rf,
        n_iter=10,
        cv=5,
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rs_rf.fit(X_train_selected, y_train)
    print(f"      Best params: {rs_rf.best_params_}")
    print(f"      Best CV F1-score: {rs_rf.best_score_:.4f}")

    best_model = rs_rf.best_estimator_
    best_model.fit(X_train_selected, y_train)

    # ---------------------------------------------------------------
    # 7. Evaluasi akhir & simpan semua artefak
    # ---------------------------------------------------------------
    print("[7/7] Evaluasi model akhir & menyimpan artefak ...")
    y_pred = best_model.predict(X_test_selected)

    metrics = {
        "model_name": "Random Forest (Tuning)",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "best_params": rs_rf.best_params_,
    }

    print("\n      === HASIL EVALUASI MODEL TERBAIK ===")
    for k in ["accuracy", "precision", "recall", "f1_score"]:
        print(f"      {k.capitalize():10s}: {metrics[k]:.4f}")

    artifacts = {
        "model": best_model,
        "scaler": scaler,
        "ordinal_encoder": oe,
        "onehot_encoder": ohe,
        "feature_columns": feature_columns,
        "top_features": top_features,
        "metrics": metrics,
    }
    output_path = "artifacts.pkl"
    joblib.dump(artifacts, output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Semua artefak berhasil digabung & disimpan ke '{output_path}' ({size_mb:.2f} MB)")
    print("   Letakkan file ini satu folder dengan app.py sebelum di-deploy.")


if __name__ == "__main__":
    main()
