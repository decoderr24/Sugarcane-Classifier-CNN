# Klasifikasi Penyakit Daun Tebu menggunakan CNN (MobileNetV2)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red.svg)

Proyek penelitian ini bertujuan untuk membangun model *deep learning* menggunakan *Convolutional Neural Network* (CNN) untuk mengklasifikasikan empat kondisi daun tebu: Sehat (*Healthy*), Busuk Merah (*Red Rot*), Karat (*Rust*), dan Daun Menguning (*Yellow Leaf*).

Aplikasi web interaktif juga dikembangkan menggunakan Streamlit untuk memungkinkan prediksi secara *real-time* melalui unggahan gambar.

![localhost_8501_ (1)](https://github.com/user-attachments/assets/8b165b32-abcc-4984-b588-5e6304a440d3)

## Demo Aplikasi
![Demo Aplikasi]

## Dataset
Dataset yang digunakan adalah **DTM1Kv1** yang tersedia di Kaggle.
* **Link Dataset:** [https://www.kaggle.com/datasets/novalsofyan/dtm1kv1/data](https://www.kaggle.com/datasets/novalsofyan/dtm1kv1/data)
* Dataset ini terdiri dari empat kelas dengan total 804 gambar yang telah dibagi menjadi data latih, validasi, dan tes.

## Struktur Proyek
```
Sugarcane-Classifier-CNN/
├── dataset/
│   └── datatebu/
│       ├── train/
│       ├── validation/
│       └── test/
├── model/
│   └── trained_model/
│       └── best_model.keras
├── result/
│   ├── training_history_plot.png
│   └── confusion_matrix_test_final.png
├── src/
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── main.py
└── requirements.txt
```

## Metodologi
Model dikembangkan menggunakan pendekatan **Transfer Learning** dengan arsitektur **MobileNetV2** sebagai *backbone* yang dibekukan (*frozen*).

1.  **Arsitektur Model**:
    * **Base Model**: MobileNetV2 (pre-trained di ImageNet) dengan lapisan atas yang tidak disertakan.
    * **Classifier Head**: Di atas *base model*, ditambahkan lapisan kustom yang terdiri dari `GlobalAveragePooling2D`, `BatchNormalization`, dua `Dense layer`, dan `Dropout` (dengan laju 0.5) untuk regularisasi yang kuat.

2.  **Augmentasi Data**:
    * Untuk mengatasi overfitting dan keterbatasan data, diterapkan strategi augmentasi data yang agresif pada data training, meliputi rotasi, pergeseran, zoom, dan perubahan kecerahan.

3.  **Proses Training**:
    * **Fase 1 (Feature Extraction)**: Melatih hanya bagian *classifier head* dengan *base model* yang sepenuhnya dibekukan.
    * **Fase 2 (Fine-Tuning)**: "Mencairkan" (*unfreezing*) sebagian kecil lapisan atas dari *base model* dan melatihnya kembali dengan *learning rate* yang sangat kecil untuk menyesuaikan fitur dengan dataset spesifik.
    * **Callbacks**: Proses training dioptimalkan dengan `ModelCheckpoint` (menyimpan model terbaik), `EarlyStopping` (menghentikan training jika tidak ada kemajuan), dan `ReduceLROnPlateau` (mengurangi *learning rate* secara otomatis).

## Hasil Akhir
Model akhir berhasil mencapai performa yang solid dalam membedakan keempat kelas penyakit.

#### Grafik Training & Validasi
![final](https://github.com/user-attachments/assets/8136d9c5-9ba1-4ac7-87cf-72e890315088)

*Grafik menunjukkan proses training yang sehat, di mana akurasi validasi terus meningkat seiring dengan penurunan loss validasi.*

#### Confusion Matrix (Data Test)
![confusion_matrix_FINAL](https://github.com/user-attachments/assets/05800923-9604-456c-8e00-f48021dd82ce)

*Confusion matrix pada data tes menunjukkan kemampuan generalisasi model yang baik pada data yang belum pernah dilihat sebelumnya.*

## Instalasi
Untuk menjalankan proyek ini secara lokal, ikuti langkah-langkah berikut:

1.  **Clone repository ini:**
    ```bash
    git clone [https://github.com/decoderr24/Sugarcane-Classifier-CNN.git](https://github.com/decoderr24/Sugarcane-Classifier-CNN.git)
    cd Sugarcane-Classifier-CNN
    ```

2.  **Buat dan aktifkan virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install semua library yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```

## Penggunaan

#### 1. Melatih Model
Untuk melatih ulang model dari awal, jalankan script berikut:
```bash
python src/train_model.py
```
*(Proses ini akan menyimpan model terbaik di folder `model/`)*

#### 2. Menjalankan Aplikasi Web
Untuk menjalankan aplikasi web interaktif dengan Streamlit, gunakan perintah:
```bash
streamlit run src/main.py
```
Buka browser Anda dan akses `http://localhost:8501`.

## Teknologi yang Digunakan
* Python
* TensorFlow & Keras
* Streamlit
* Scikit-learn
* Seaborn & Matplotlib
* NumPy
