import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# =================================================================================
# KONFIGURASI HALAMAN DAN JUDUL
# =================================================================================
st.set_page_config(
    page_title="Deteksi Penyakit Tebu",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Deteksi Penyakit Daun Tebu")
st.write(
    "Unggah gambar daun tebu untuk diklasifikasikan oleh model CNN. "
    "Model akan memprediksi apakah daun tersebut Sehat, Red Rot, Rust, atau Yellow Leaf."
)

# =================================================================================
# FUNGSI UNTUK MEMUAT MODEL DAN PREDIKSI
# =================================================================================

# Fungsi untuk memuat model (menggunakan cache agar tidak loading berulang)
@st.cache_resource
def load_trained_model():
    """Memuat model Keras terbaik yang sudah dilatih."""
    MODEL_PATH = os.path.join('model', 'trained_model', 'best_model.keras')
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# Fungsi untuk memproses gambar dan melakukan prediksi
def predict_image(model, image_to_predict):
    """Memproses gambar dan melakukan prediksi."""
    class_names = ['healthy', 'redrot', 'rust', 'yellow']
    
    # Pra-pemrosesan gambar
    img = image_to_predict.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Buat batch
    
    # Prediksi
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    return predicted_class, confidence

# =================================================================================
# TAMPILAN UTAMA APLIKASI
# =================================================================================

# Muat model
model = load_trained_model()

if model:
    # Widget untuk mengunggah file gambar
    uploaded_file = st.file_uploader(
        "Pilih sebuah gambar...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        
        # Buat dua kolom untuk tata letak yang lebih baik
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Gambar yang Diunggah', use_column_width=True)
        
        # Tombol untuk memulai prediksi
        if st.button('Analisis Gambar'):
            with st.spinner('Model sedang bekerja...'):
                predicted_class, confidence = predict_image(model, image)
                
                with col2:
                    st.success("Analisis Selesai!")
                    st.metric(label="Hasil Prediksi", value=predicted_class.capitalize())
                    st.write(f"**Tingkat Keyakinan:** {confidence:.2f}%")
                    
                    # Berikan deskripsi singkat tentang penyakit
                    if predicted_class == 'redrot':
                        st.error("Deskripsi: Red Rot ditandai dengan bercak merah memanjang pada tulang daun.")
                    elif predicted_class == 'rust':
                        st.warning("Deskripsi: Rust ditandai dengan bintik-bintik kecil berwarna oranye hingga coklat.")
                    elif predicted_class == 'yellow':
                        st.warning("Deskripsi: Yellow Leaf ditandai dengan menguningnya daun yang dimulai dari ujung.")
                    else:
                        st.info("Deskripsi: Tanaman terdeteksi dalam kondisi sehat.")
else:
    st.header("Gagal memuat model. Pastikan file 'best_model.keras' ada di dalam folder 'model'.")