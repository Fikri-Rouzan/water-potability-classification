# HydroCheck

## 📌 Deskripsi

Proyek ini merupakan implementasi pemodelan machine learning yang digunakan untuk mengklasifikasikan tingkat kelayakan konsumsi air berdasarkan parameter kualitas fisik dan kimia. Pemodelan dibangun dengan menganalisis sampel data yang memuat atribut spesifik kualitas air untuk memprediksi status kelayakan konsumsi. Hasil dari proyek ini ditujukan sebagai alat bantu dalam memantau kondisi sumber daya air guna mendukung penyediaan air bersih yang memenuhi standar keamanan konsumsi manusia.

---

## 💾 Dataset

Dataset yang digunakan dalam proyek ini bersumber dari [Kaggle: Water Quality and Potability](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability). Data ini memuat matriks pengukuran kualitas air dan penilaian terkait kelayakannya untuk dikonsumsi oleh manusia. Di dalamnya terdapat sejumlah baris data sampel air yang dilengkapi dengan berbagai parameter fisikokimia serta kolom indikator kelayakan untuk mempermudah analisis kecocokan air terhadap kebutuhan konsumsi harian.

---

## 🛠️ Tech Stack

| Kategori                    | Teknologi yang Digunakan                                    |
| :-------------------------- | :---------------------------------------------------------- |
| 🌐 **Programming Language** | `Python`                                                    |
| 🌱 **Environment**          | `Jupyter Notebook`                                          |
| 🧩 **Framework**            | `Streamlit`                                                 |
| ⚛️ **Libraries**            | `pandas`, `Matplotlib`, `seaborn`, `scikit-learn`, `Joblib` |
| ⚡ **Tool**                 | `Google Colab`                                              |
| 🚀 **Deployment**           | `Streamlit Community Cloud`                                 |

---

## ⚙️ Petunjuk Pengaturan

1. **Prasyarat**
   - Python 3.11 atau lebih baru.
   - Git terinstal di komputer.

2. **Clone Repositori**

```bash
git clone https://github.com/Fikri-Rouzan/hydrocheck.git
cd hydrocheck
```

3. **Buat Virtual Environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

4. **Install Dependensi**

```bash
pip install -r requirements.txt
```

5. **Menjalankan Dashboard Streamlit**

```bash
streamlit run dashboard/dashboard.py
```
