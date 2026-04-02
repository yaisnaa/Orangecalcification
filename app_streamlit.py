import streamlit as st
import pandas as pd
import joblib
st.set_page_config(
    page_title= "Klasifikasi jeruk"
)
st.title = ":tangerine: Klasifikasi jeruk"
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="font-size: 45px; color: #ff4b4b; margin-bottom: 10px;">
            Orange Quality Predictor
        </h1>
        <p style="font-size: 20px; color: #555555; font-style: italic;">
            Gunakan aplikasi ini untuk memprediksi kualitas jeruk secara instan berdasarkan data fisik jeruk.
        </p>
        <hr style="border: 0; height: 1px; background: #eee; margin-bottom: 30px;">
    </div>
    """, 
    unsafe_allow_html=True
)
df = pd.read_csv("jeruk_balance_500.csv")
model = joblib.load("model_klasifikasi_jeruk.joblib")

diameter = st.slider("Diameter",1.0, 10.0, 6.5)
berat = st.slider("Berat", 0.1, 300.0, 150.0)
tebal_kulit = st.slider("Tebal kulit", 0.1, 2.0, 1.0)
kadar_gula = st.slider("Kadar gula", 0.1, 15.0, 7.0)
asal_daerah = st.pills("Asal daerah",["Kalimantan", "Jawa Barat", "Jawa Tengah"])
warna = st.pills("Warna", ["hijau", "kuning", "oranye"])
musim_panen = st.pills("Musim panen",["kemarau", "hujan"])

if st.button("Prediksi", type="primary"):
    if(asal_daerah is None or  warna is None or musim_panen is None):
        st.error("***Data yang dimasukan belum lengkap, harap masukan data dengan lengkap***")
    else:
        data_baru = pd.DataFrame([[diameter, berat, tebal_kulit, kadar_gula, asal_daerah,warna, musim_panen]], columns=["diameter","berat","tebal_kulit", "kadar_gula","asal_daerah","warna", "musim_panen"])
        hasil = model.predict(data_baru)[0]
        prob = max(model.predict_proba(data_baru)[0])
        if prob < 0.7:
            st.error("Kebenaran hasil prediksi **< 70%**, sistem tidak bisa memberikan hasil")
        else:
            st.success(f"Prediksi menunjukan kualitas: **{hasil}** \n dengan probabilitas kebenaran: **{prob*100:.2f}%**")    