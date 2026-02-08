import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Sugarcane Classifier", layout="centered")

@st.cache_resource
def load_saved_model():
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))
        path_to_load = os.path.join(project_root, "model", "best_model_rebuild.h5")
        
        if not os.path.exists(path_to_load):
            st.error(f"❌ File model tidak ditemukan di: {path_to_load}")
            return None

        st.info("🚀 Memuat model dengan mode kompatibilitas...")
        
        # TEKNIK BARU: Muat tanpa kompilasi DAN gunakan custom_objects kosong
        # Ini akan memaksa Keras mengabaikan parameter layer yang tidak dikenal
        model = tf.keras.models.load_model(
            path_to_load, 
            compile=False,
            custom_objects=None,
            safe_mode=False # Khusus untuk versi TensorFlow terbaru
        )
        
        st.success("✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_saved_model()
class_names = ['healthy', 'redrot', 'rust', 'yellow']

st.title("🌿 Sugarcane Leaf Disease Classifier")
st.markdown("Mode: **Keras H5 Model**")

uploaded_file = st.file_uploader("📤 Upload gambar daun tebu", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼️ Gambar yang diupload", use_container_width=True)

    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Prediksi untuk model .h5 lebih sederhana dibanding SavedModel
    predictions = model.predict(img_batch)
    
    pred_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    predicted_label = class_names[pred_index]

    # --- BAGIAN OUTPUT ---
    st.markdown("### 🧠 Hasil Prediksi:")
    st.success(f"Kelas Prediksi: **{predicted_label}**")
    st.info(f"Keyakinan: **{confidence:.2f}%**")

    # Menampilkan probabilitas semua kelas
    st.markdown("### 📊 Rincian Probabilitas:")
    for i, class_name in enumerate(class_names):
        probability = predictions[0][i] * 100
        st.write(f"- {class_name}: {probability:.2f}%")
