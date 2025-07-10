import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# -------------------------------
# Konfigurasi Halaman Streamlit
# -------------------------------
st.set_page_config(page_title="Klasterisasi Motor", layout="centered")

st.title("Aplikasi Klasterisasi Sepeda Motor")
st.write("Masukkan data motor untuk mengetahui klasternya")

# ---------------------------------
# Form Input Data dari Pengguna
# ---------------------------------
with st.form("input_form"):
    merek = st.text_input("Merek Motor")
    harga = st.number_input("Harga (Rp)", min_value=0, step=1000000)
    kapasitas_mesin = st.number_input("Kapasitas Mesin (cc)", min_value=0)
    berat = st.number_input("Berat (kg)", min_value=0)
    konsumsi_bbm = st.number_input("Konsumsi BBM (km/liter)", min_value=0)
    transmisi = st.selectbox("Jenis Transmisi", ["Manual", "Otomatis"])

    submitted = st.form_submit_button("Proses Klaster")
# ---------------------------------
# Proses Saat Tombol Submit
# ---------------------------------
if submitted:
    # Memuat scaler dan model yang telah disimpan sebelumnya
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("kmeans_model.pkl")

    # Kolom sesuai yang digunakan saat training
    fitur = ['Harga (Juta)', 'Kapasitas Mesin (cc)', 'Transmisi', 'Konsumsi BBM (km/l)', 'Berat (kg)']

    # Mapping transmisi
    transmisi_value = 0 if transmisi == "Manual" else 1

    # Buat input data sesuai urutan kolom saat training
    data_input = pd.DataFrame([[harga, kapasitas_mesin, transmisi_value, konsumsi_bbm, berat]], columns=fitur)

    # Scaling
    data_scaled = scaler.transform(data_input)

    # Prediksi
    klaster = model.predict(data_scaled)[0]

    st.success(f"Motor **{merek}** termasuk dalam **Klaster ke-{klaster}**")

