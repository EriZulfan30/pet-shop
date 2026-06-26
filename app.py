import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =====================================================================
# KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(page_title="Sistem Prediksi Stok Petshop", page_icon="🐾", layout="wide")

st.title("🐾 Sistem Pendukung Keputusan Prediksi Stok - Cipanas Pets Shop")
st.markdown("""
Aplikasi ini memprediksi kebutuhan stok produk menggunakan algoritma **Multiple Linear Regression** berdasarkan kerangka kerja **CRISP-DM**.
Model cerdas ini telah dilatih sebelumnya dan siap digunakan untuk memproyeksikan kebutuhan stok harian.
""")

# =====================================================================
# LOAD MODEL & SCALER DARI GITHUB (.pkl)
# =====================================================================
@st.cache_resource
def load_machine_learning_assets():
    try:
        # Membaca file .pkl yang sudah di-upload ke GitHub
        loaded_model = joblib.load('model_regresi_cipanas.pkl')
        loaded_scaler = joblib.load('scaler_cipanas.pkl')
        return loaded_model, loaded_scaler, None
    except Exception as e:
        return None, None, str(e)

model, scaler, error_msg = load_machine_learning_assets()

if error_msg:
    st.error(f"⚠️ Gagal memuat file kecerdasan buatan. Pastikan file `model.pkl` dan `scaler.pkl` sudah diunggah ke GitHub! Error detail: {error_msg}")
    st.stop()

# =====================================================================
# BAGIAN 1: INPUT MANUAL (SIMULASI HARIAN)
# =====================================================================
st.header("🔧 Simulasi Kebutuhan Stok Harian")
st.write("Gunakan panel ini untuk mengecek kebutuhan stok produk untuk hari esok secara cepat.")

col1, col2, col3 = st.columns(3)

with col1:
    tren_input = st.number_input("Rata-rata Penjualan 7 Hari Terakhir (Unit)", min_value=0.0, value=15.0, step=1.0)
with col2:
    is_weekend = st.selectbox("Apakah besok Akhir Pekan? (Sabtu/Minggu)", ["Tidak", "Ya"])
with col3:
    is_payday = st.selectbox("Apakah besok Periode Gajian? (Tgl 25 - 1)", ["Tidak", "Ya"])

if st.button("Hitung Prediksi Stok", use_container_width=True):
    # Konversi String ke Biner (1/0)
    weekend_val = 1 if is_weekend == "Ya" else 0
    payday_val = 1 if is_payday == "Ya" else 0
    
    # Format input sesuai yang diharapkan scaler
    input_data = np.array([[tren_input, weekend_val, payday_val]])
    
    # Lakukan scaling dan prediksi
    input_scaled = scaler.transform(input_data)
    hasil_prediksi = model.predict(input_scaled)[0]
    
    # Amankan hasil agar tidak minus dan bulatkan
    stok_rekomendasi = max(0, int(np.round(hasil_prediksi)))
    
    st.success(f"📦 **Estimasi Kebutuhan Stok:** {stok_rekomendasi} Unit")


st.divider()

# =====================================================================
# BAGIAN 2: UPLOAD DATASET UNTUK EVALUASI MASSAL (GRAFIK)
# =====================================================================
st.header("📊 Evaluasi Kinerja Algoritma (Batch Prediction)")
st.write("Silakan unggah database log transaksi terakhir (Excel) untuk mengevaluasi akurasi model secara massal.")

uploaded_file = st.file_uploader("Upload File Dataset Excel (.xlsx)", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Membaca data dan pra-pemrosesan
        df_raw = pd.read_excel(uploaded_file, sheet_name='Riwayat Transaksi', header=2)
        df_sales = df_raw[df_raw['Jenis Transaksi'] == 'Penjualan'].copy()
        df_sales['Tanggal'] = pd.to_datetime(df_sales['Tanggal'], dayfirst=True, errors='coerce')
        
        df_daily = df_sales.groupby('Tanggal').agg({
            'Jumlah (Unit)': 'sum',
            'Akhir Pekan': 'first',
            'Hari Gajian': 'first'
        }).reset_index()

        df_daily['is_weekend'] = df_daily['Akhir Pekan'].apply(lambda x: 1 if str(x).strip() == 'Ya' else 0)
        df_daily['is_payday'] = df_daily['Hari Gajian'].apply(lambda x: 1 if str(x).strip() == 'Ya' else 0)
        df_daily['tren_penjualan_7hari'] = df_daily['Jumlah (Unit)'].shift(1).rolling(window=7).mean()
        df_daily = df_daily.dropna()
        
        # Eksekusi Prediksi Massal menggunakan Scaler dan Model dari .pkl
        X_batch = df_daily[['tren_penjualan_7hari', 'is_weekend', 'is_payday']]
        Y_aktual = df_daily['Jumlah (Unit)']
        
        X_batch_scaled = scaler.transform(X_batch)
        Y_pred_batch = model.predict(X_batch_scaled)
        
        Y_pred_batch = np.where(Y_pred_batch < 0, 0, Y_pred_batch)
        Y_pred_rounded = np.round(Y_pred_batch).astype(int)
        
        df_daily['Prediksi Sistem'] = Y_pred_rounded
        
        # Tampilkan Grafik
        st.subheader("Grafik Aktual vs Prediksi")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_daily['Tanggal'], Y_aktual, label='Kebutuhan Aktual', marker='o', color='royalblue')
        ax.plot(df_daily['Tanggal'], Y_pred_rounded, label='Prediksi Algoritma ML', marker='s', color='crimson', alpha=0.8)
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Unit Produk")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
        
        # Tampilkan Tabel
        st.subheader("📋 Log Tabel Perbandingan")
        tabel_tampil = df_daily[['Tanggal', 'Jumlah (Unit)', 'Prediksi Sistem', 'Akhir Pekan', 'Hari Gajian']].tail(10)
        tabel_tampil['Tanggal'] = tabel_tampil['Tanggal'].dt.strftime('%d-%m-%Y')
        tabel_tampil.rename(columns={'Jumlah (Unit)': 'Aktual Terjual'}, inplace=True)
        st.dataframe(tabel_tampil, use_container_width=True)
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file Excel: {e}")
