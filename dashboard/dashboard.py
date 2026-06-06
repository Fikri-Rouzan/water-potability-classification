import streamlit as st
import pandas as pd
import os
import joblib
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

# Memastikan komponen model termuat
if scaler and rf_model and svm_model:
    # Tab 1 untuk prediksi kelayakan air
    with tab1:
        with st.container(border=True):
            st.subheader("Parameter Air")

            col1, col2, col3 = st.columns(3)
            with col1:
                ph = st.slider(
                    "pH (Tingkat pH)",
                    min_value=0.0,
                    max_value=14.0,
                    value=7.0,
                    step=0.1,
                )
                hardness = st.slider(
                    "Hardness (Kesadahan Air)",
                    min_value=0.0,
                    max_value=400.0,
                    value=185.0,
                    step=1.0,
                )
                solids = st.slider(
                    "Solids / TDS (Total Padatan Terlarut)",
                    min_value=0.0,
                    max_value=50000.0,
                    value=14800.0,
                    step=100.0,
                )
            with col2:
                chloramines = st.slider(
                    "Chloramines (Kloramin)",
                    min_value=0.0,
                    max_value=15.0,
                    value=8.8,
                    step=0.1,
                )
                sulfate = st.slider(
                    "Sulfate (Sulfat)",
                    min_value=0.0,
                    max_value=600.0,
                    value=444.0,
                    step=1.0,
                )
                conductivity = st.slider(
                    "Conductivity (Konduktivitas)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=282.0,
                    step=1.0,
                )
            with col3:
                organic_carbon = st.slider(
                    "Organic Carbon (Karbon Organik)",
                    min_value=0.0,
                    max_value=40.0,
                    value=19.8,
                    step=0.1,
                )
                trihalomethanes = st.slider(
                    "Trihalomethanes (Trihalometana)",
                    min_value=0.0,
                    max_value=150.0,
                    value=75.1,
                    step=0.1,
                )
                turbidity = st.slider(
                    "Turbidity (Kekeruhan)",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.9,
                    step=0.1,
                )

        # Parameter input untuk prediksi
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
        features_scaled_df = pd.DataFrame(features_scaled, columns=feature_names)

        # Memilih model berdasarkan pilihan di sidebar
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

        # Hasil analisis dan rekomendasi
        st.write("")
        st.subheader("Hasil Analisis")

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

        st.markdown("---")

        # Tombol untuk menyimpan hasil prediksi ke riwayat
        if st.button(
            "💾 Simpan Hasil Prediksi",
            type="primary",
            use_container_width=True,
        ):
            timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
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
            st.toast("Prediksi berhasil disimpan ke tab 'Riwayat'!", icon="💾")

else:
    st.error("Model belum dimuat dengan benar. Pastikan komponen model tersedia.")

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
            "Belum ada riwayat prediksi. Silakan tekan tombol 'Simpan Hasil Prediksi' di tab 'Prediksi Kelayakan'."
        )
    else:
        st.write(f"Total pengujian pada sesi ini: **{total_items} data**")

        # Mengonversi list riwayat menjadi datagrame
        history_df = pd.DataFrame(
            [
                {
                    "Waktu": item["time"],
                    "Model": item["model"],
                    "Status": item["status"],
                    "pH": item["params"]["pH"],
                    "Hardness": item["params"]["Hardness"],
                    "Solids (TDS)": item["params"]["Solids"],
                    "Chloramines": item["params"]["Chloramines"],
                    "Sulfate": item["params"]["Sulfate"],
                    "Conductivity": item["params"]["Conductivity"],
                    "Organic Carbon": item["params"]["Organic Carbon"],
                    "Trihalomethanes": item["params"]["Trihalomethanes"],
                    "Turbidity": item["params"]["Turbidity"],
                }
                for item in st.session_state.history
            ]
        )

        # Mengonversi dataframe menjadi CSV
        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Unduh Seluruh Riwayat (CSV)",
            data=csv_data,
            file_name=f"riwayat_analisis_air_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("---")

        # Pengaturan pagination
        ITEMS_PER_PAGE = 5
        total_pages = math.ceil(total_items / ITEMS_PER_PAGE)

        # Jika ada lebih dari 1 halaman, tampilkan button untuk navigasi halaman
        if total_pages > 1:
            current_page = st.radio(
                "Pilih Halaman:",
                options=range(1, total_pages + 1),
                horizontal=True,
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
                    st.write(
                        f"**🕒 Waktu Penyimpanan:** {record['time']} | Model: `{record['model']}`"
                    )
                    st.write(f"{record['icon']} **{record['status']}**")

                    # Menampilkan detail parameter input
                    if "params" in record:
                        with st.expander("Lihat Detail Parameter Input"):
                            p = record["params"]
                            c1, c2, c3 = st.columns(3)

                            with c1:
                                st.caption(f"pH: {p['pH']}")
                                st.caption(f"Hardness: {p['Hardness']}")
                                st.caption(f"Solids: {p['Solids']}")
                            with c2:
                                st.caption(f"Chloramines: {p['Chloramines']}")
                                st.caption(f"Sulfate: {p['Sulfate']}")
                                st.caption(f"Conductivity: {p['Conductivity']}")
                            with c3:
                                st.caption(f"Organic Carbon: {p['Organic Carbon']}")
                                st.caption(f"Trihalomethanes: {p['Trihalomethanes']}")
                                st.caption(f"Turbidity: {p['Turbidity']}")

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
