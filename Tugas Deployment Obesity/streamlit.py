
import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_artefak():
    scaler = joblib.load("scaler_obesity.joblib")
    model_kmeans = joblib.load("kmeans_obesity.joblib")
    return scaler, model_kmeans

scaler, model_kmeans = load_artefak()

st.title("🧍 Segmentasi Individu dengan K-Means")
st.write("Aplikasi untuk memprediksi cluster berdasarkan kondisi fisik dan kebiasaan.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", value=25.0)
    height = st.number_input("Height", value=1.70)
    weight = st.number_input("Weight", value=70.0)
    fcvc = st.number_input("FCVC", value=2.0)

with col2:
    ncp = st.number_input("NCP", value=3.0)
    ch2o = st.number_input("CH2O", value=2.0)
    faf = st.number_input("FAF", value=1.0)
    tue = st.number_input("TUE", value=1.0)

if st.button("Prediksi Cluster"):

    input_data = pd.DataFrame(
        [[
            age,
            height,
            weight,
            fcvc,
            ncp,
            ch2o,
            faf,
            tue
        ]],
        columns=[
            "Age",
            "Height",
            "Weight",
            "FCVC",
            "NCP",
            "CH2O",
            "FAF",
            "TUE"
        ]
    )

    scaled_input = scaler.transform(input_data)

    cluster_result = model_kmeans.predict(
        scaled_input
    )[0]

    st.success(
        f"Individu ini masuk ke dalam: **Cluster {cluster_result}**"
    )

    st.metric(
        label="Status Prediksi",
        value="Berhasil",
        delta="Selesai"
    )
