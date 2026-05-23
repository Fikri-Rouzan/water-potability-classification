import streamlit as st
import pandas as pd
import os
import joblib
import time
import math
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(page_title="HydroCheck Analytics", page_icon="💧", layout="wide")

# Session State untuk menyimpan riwayat prediksi
if "history" not in st.session_state:
    st.session_state.history = []


def delete_history(index):
    st.session_state.history.pop(index)


# Memuat model dengan caching
@st.cache_resource
def load_models():
    MODEL_DIR = "models"

    try:
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
        svm_model = joblib.load(os.path.join(MODEL_DIR, "svm_model.joblib"))
        return scaler, rf_model, svm_model
    except Exception as e:
        st.error(f"Gagal memuat model. Error: {e}")
        return None, None, None


scaler, rf_model, svm_model = load_models()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3105/3105807.png", width=120)
    st.title("Data Kualitas Air")
    st.markdown("---")

    model_choice = st.selectbox(
        label="Pilih Model Klasifikasi",
        options=("🌲 Random Forest", "📊 Support Vector Machine (SVM)"),
    )

    st.markdown("---")
    st.caption(
        "Dashboard ini dibuat untuk menganalisis parameter fisik dan kimia air "
        "guna memprediksi kelayakan air minum menggunakan algoritma Machine Learning."
    )

# UI Utama
st.title("Dashboard HydroCheck: Analisis Kualitas Air 💧")
st.markdown(
    "Menampilkan insight dan prediksi kelayakan konsumsi air minum berdasarkan input parameter laboratorium."
)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    ["🔍 Prediksi Kelayakan", "📖 Panduan Parameter", "🕒 Riwayat"]
)

# Tab 1 untuk prediksi kelayakan air
with tab1:
    with st.container(border=True):
        st.subheader("Input Parameter Air")

        col1, col2, col3 = st.columns(3)
        with col1:
            ph = st.number_input(
                "pH (0-14)", min_value=0.0, max_value=14.0, value=7.0, step=0.1
            )
            chloramines = st.number_input(
                "Chloramines", min_value=0.0, value=7.4, step=0.1
            )
            organic_carbon = st.number_input(
                "Organic Carbon", min_value=0.0, value=19.8, step=0.1
            )
        with col2:
            hardness = st.number_input("Hardness", min_value=0.0, value=185.0, step=1.0)
            sulfate = st.number_input("Sulfate", min_value=0.0, value=367.0, step=1.0)
            trihalomethanes = st.number_input(
                "Trihalomethanes", min_value=0.0, value=75.1, step=0.1
            )
        with col3:
            solids = st.number_input(
                "Solids (TDS)", min_value=0.0, value=14800.0, step=100.0
            )
            conductivity = st.number_input(
                "Conductivity", min_value=0.0, value=435.0, step=1.0
            )
            turbidity = st.number_input("Turbidity", min_value=0.0, value=1.9, step=0.1)

    st.write("")

    # Button untuk menjalankan prediksi
    if st.button("🔍︎ Analisis Sampel Air", type="primary", use_container_width=True):
        if scaler and rf_model and svm_model:
            with st.spinner("Menganalisis data..."):
                time.sleep(0.5)

                input_features = [
                    ph,
                    hardness,
                    solids,
                    chloramines,
                    sulfate,
                    conductivity,
                    organic_carbon,
                    trihalomethanes,
                    turbidity,
                ]

                feature_names = [
                    "ph",
                    "Hardness",
                    "Solids",
                    "Chloramines",
                    "Sulfate",
                    "Conductivity",
                    "Organic_carbon",
                    "Trihalomethanes",
                    "Turbidity",
                ]

                features_df = pd.DataFrame([input_features], columns=feature_names)
                features_scaled = scaler.transform(features_df)
                features_scaled_df = pd.DataFrame(
                    features_scaled, columns=feature_names
                )

                # Memilih model berdasarkan pilihan
                if "SVM" in model_choice:
                    prediction = svm_model.predict(features_scaled_df)
                    model_used = "SVM"
                else:
                    prediction = rf_model.predict(features_scaled_df)
                    model_used = "Random Forest"

                # Menentukan status berdasarkan hasil prediksi
                if prediction[0] == 1:
                    status_text = "LAYAK MINUM (POTABLE)"
                    status_icon = "✅"
                    alert_type = "success"
                else:
                    status_text = "TIDAK LAYAK MINUM (NOT POTABLE)"
                    status_icon = "⚠️"
                    alert_type = "error"

                timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

                # Menyimpan hasil prediksi ke dalam session state history
                st.session_state.history.append(
                    {
                        "time": timestamp,
                        "model": model_used,
                        "status": status_text,
                        "icon": status_icon,
                        "params": {
                            "pH": ph,
                            "Hardness": hardness,
                            "Solids": solids,
                            "Chloramines": chloramines,
                            "Sulfate": sulfate,
                            "Conductivity": conductivity,
                            "Organic Carbon": organic_carbon,
                            "Trihalomethanes": trihalomethanes,
                            "Turbidity": turbidity,
                        },
                    }
                )

                st.markdown("---")
                st.subheader("Hasil Analisis")

                # Menampilkan hasil berdasarkan alert
                if alert_type == "success":
                    st.success(f"### {status_icon} STATUS: {status_text}")
                    st.write(
                        f"Berdasarkan klasifikasi algoritma **{model_used}**, karakteristik air ini memenuhi standar keamanan dasar."
                    )
                else:
                    st.error(f"### {status_icon} STATUS: {status_text}")
                    st.write(
                        f"Berdasarkan klasifikasi algoritma **{model_used}**, air ini memiliki parameter yang beresiko jika dikonsumsi."
                    )
        else:
            st.error("Model belum dimuat dengan benar. Pastikan folder 'models' ada.")

# Tab 2 untuk panduan parameter
with tab2:
    st.subheader("Panduan Parameter")
    st.write("Berikut adalah penjelasan untuk setiap parameter pengukuran air:")

    # Parameter air
    with st.expander("pH (Tingkat pH)"):
        st.info(
            "Mengukur tingkat keasaman atau kebasaan air. Air murni memiliki pH 7. WHO merekomendasikan pH air minum berada di antara 6,5 hingga 8,5."
        )
    with st.expander("Hardness (Kesadahan Air)"):
        st.info(
            "Kapasitas air untuk mengendapkan sabun, yang disebabkan oleh kandungan mineral seperti kalsium dan magnesium."
        )
    with st.expander("Solids / TDS (Total Padatan Terlarut)"):
        st.info(
            "Jumlah material organik dan anorganik yang tersuspensi di dalam air. Parameter ini menunjukkan tingginya tingkat mineralisasi pada air."
        )
    with st.expander("Chloramines (Kloramin)"):
        st.info(
            "Konsentrasi senyawa kloramin di dalam air. Senyawa ini sering digunakan sebagai desinfektan dalam sistem pengolahan air minum publik."
        )
    with st.expander("Sulfate (Sulfat)"):
        st.info(
            "Konsentrasi mineral sulfat alami di dalam air. Konsentrasi yang terlalu tinggi dapat memberikan rasa pahit dan efek laksatif (pencahar)."
        )
    with st.expander("Conductivity (Konduktivitas)"):
        st.info(
            "Tingkat kemampuan air dalam menghantarkan arus listrik. Angka ini berhubungan langsung dengan tingginya konsentrasi ion yang terlarut di dalamnya."
        )
    with st.expander("Organic Carbon (Karbon Organik)"):
        st.info(
            "Jumlah total karbon dalam senyawa organik yang terkandung di dalam air. Indikator ini sering digunakan untuk mengevaluasi kualitas air secara umum."
        )
    with st.expander("Trihalomethanes (Trihalometana)"):
        st.info(
            "Bahan kimia yang dapat terbentuk ketika klorin bereaksi dengan materi organik alami yang ada di dalam air. Kadarnya dibatasi untuk alasan kesehatan."
        )
    with st.expander("Turbidity (Kekeruhan)"):
        st.info(
            "Ukuran kejernihan air. Kekeruhan diakibatkan oleh partikel tersuspensi yang tidak kasat mata, dan ini merupakan tes utama dalam kualitas air."
        )

# Tab 3 untuk riwayat prediksi
with tab3:
    st.subheader("Riwayat Prediksi")

    total_items = len(st.session_state.history)

    if total_items == 0:
        st.info(
            "Belum ada riwayat prediksi. Silakan lakukan analisis terlebih dahulu di tab 'Prediksi Kelayakan'."
        )
    else:
        st.write(f"Total pengujian pada sesi ini: **{total_items} data**")

        # Pengaturan pagination
        ITEMS_PER_PAGE = 5
        total_pages = math.ceil(total_items / ITEMS_PER_PAGE)

        # Jika ada lebih dari 1 halaman, tampilkan button untuk navigasi halaman
        if total_pages > 1:
            current_page = st.radio(
                "Pilih Halaman:", options=range(1, total_pages + 1), horizontal=True
            )
        else:
            current_page = 1

        # Hitung index awal dan akhir untuk slicing data
        start_idx = (current_page - 1) * ITEMS_PER_PAGE
        end_idx = start_idx + ITEMS_PER_PAGE

        # Menampilkan data sesuai dengan halaman yang dipilih
        for i in range(start_idx, min(end_idx, total_items)):
            record = st.session_state.history[i]

            with st.container(border=True):
                col_text, col_btn = st.columns([5, 1])

                with col_text:
                    st.write(f"**{record['time']}** | Model: `{record['model']}`")
                    st.write(f"{record['icon']} **{record['status']}**")

                    p = record["params"]
                    st.caption(
                        f"**pH:** {p['pH']} | **Hardness:** {p['Hardness']} | **Solids:** {p['Solids']} | "
                        f"**Chloramines:** {p['Chloramines']} | **Sulfate:** {p['Sulfate']} | "
                        f"**Conductivity:** {p['Conductivity']} | **Org. Carbon:** {p['Organic Carbon']} | "
                        f"**Trihalomethanes:** {p['Trihalomethanes']} | **Turbidity:** {p['Turbidity']}"
                    )

                # Button untuk menghapus riwayat prediksi
                with col_btn:
                    st.button(
                        "🗑️ Hapus",
                        key=f"delete_btn_{i}_{record['time']}",
                        on_click=delete_history,
                        args=(i,),
                        use_container_width=True,
                    )

# Footer
st.markdown("---")
st.caption("© 2026 Muhammad Fikri Rouzan Ash Shidik")
