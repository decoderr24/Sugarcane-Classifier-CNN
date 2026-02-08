import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Sugarcane Classifier", layout="centered")

@st.cache_resource
def load_saved_model():
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))
        path_to_load = os.path.join(project_root, "model", "best_model_rebuild.h5")
        
        if not os.path.exists(path_to_load):
            st.error(f"❌ File model tidak ditemukan!")
            return None

        st.info("🚀 Memuat model dalam Mode Legacy...")
        
        # Menggunakan jalur legacy untuk menghindari error DTypePolicy dan batch_shape
        from tensorflow.keras.layers import InputLayer
        import tensorflow.keras as keras

        # Trik untuk mengabaikan metadata yang tidak dikenali
        model = tf.keras.models.load_model(
            path_to_load, 
            compile=False,
            custom_objects={'InputLayer': InputLayer, 'DTypePolicy': lambda **kwargs: None}
        )
        
        st.success("✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None
# Inisialisasi model dan daftar kelas
model = load_saved_model()
class_names = ['healthy', 'redrot', 'rust', 'yellow']

st.title("🌿 Sugarcane Leaf Disease Classifier")
st.markdown("Mode: **Keras H5 Model (Compatibility Mode)**")

# Widget upload gambar
uploaded_file = st.file_uploader("📤 Upload gambar daun tebu", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Memproses gambar
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼️ Gambar yang diupload", use_container_width=True)

    # Preprocessing: Resize dan Normalisasi
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Melakukan Prediksi
    with st.spinner('Sedang menganalisis...'):
        predictions = model.predict(img_batch)
    
    pred_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    predicted_label = class_names[pred_index]

    # --- TAMPILAN HASIL ---
    st.markdown("---")
    st.markdown("### 🧠 Hasil Prediksi:")
    
    # Memberi warna berbeda berdasarkan hasil
    if predicted_label == 'healthy':
        st.balloons()
        st.success(f"Kondisi: **SEHAT** ({predicted_label})")
    else:
        st.warning(f"Terdeteksi Penyakit: **{predicted_label.upper()}**")
        
    st.info(f"Tingkat Keyakinan: **{confidence:.2f}%**")

    # Rincian probabilitas tiap kelas
    with st.expander("📊 Lihat Rincian Probabilitas"):
        for i, class_name in enumerate(class_names):
            probability = predictions[0][i] * 100
            st.write(f"- **{class_name}**: {probability:.2f}%")
