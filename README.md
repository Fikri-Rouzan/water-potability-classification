# Water Potability Classification

## 📌 Deskripsi

Proyek ini dirancang untuk memetakan dan mengklasifikasikan tingkat kelayakan konsumsi air berdasarkan berbagai parameter kualitas fisik dan kimia yang terkandung di dalamnya. Dengan memanfaatkan pemodelan berbasis machine learning, sistem ini mampu memprediksi status kelayakan air berdasarkan sampel data kualitas yang dimasukkan. Hasil akhir ini bertujuan memberikan wawasan mendalam mengenai pemantauan kondisi sumber air guna mendukung penyediaan air bersih yang aman bagi kebutuhan konsumsi manusia.

---

## 💾 Dataset

Dataset yang digunakan dalam proyek ini bersumber dari [Kaggle: Water Quality and Potability](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability). Dataset ini menyajikan gambaran mendalam mengenai pengukuran kualitas air dan penilaian terkait kelayakannya untuk dikonsumsi oleh manusia. Di dalamnya mencakup berbagai baris data yang mewakili sampel air dengan atribut parameter spesifik, yang dilengkapi dengan indikator kelayakan untuk membantu menentukan serta menganalisis apakah air tersebut aman atau tidak untuk digunakan sehari-hari.

---

## 🛠️ Tech Stack

| Kategori                    | Teknologi yang Digunakan                                     |
| :-------------------------- | :----------------------------------------------------------- |
| 🌐 **Programming Language** | `Python`                                                     |
| 🌱 **Environment**          | `Jupyter Notebook`                                           |
| 🧩 **Framework**            | `Streamlit`                                                  |
| ⚛️ **Libraries**            | `pandas`, `Matplotlib`, `seaborn`, `scikit-learn`, `Joblib` |
| ⚡ **Tool**                 | `Google Colab`                                               |
| 🚀 **Deployment**           | `Streamlit Community Cloud`                                  |

---

## ⚙️ Petunjuk Pengaturan

1. **Prasyarat**
   - Python 3.11 atau lebih baru.
   - Git terinstal di komputer.

2. **Clone Repositori**

```bash
git clone https://github.com/Fikri-Rouzan/water-potability-classification.git
cd water-potability-classification
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
