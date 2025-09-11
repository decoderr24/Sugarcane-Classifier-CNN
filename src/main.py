import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Sugarcane Classifier", layout="centered")

@st.cache_resource
def load_saved_model():
    """Memuat TensorFlow SavedModel dari direktori lokal menggunakan path absolut."""
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))
        path_to_load = os.path.join(project_root, "model", "best_model_export")
        
        if not os.path.exists(path_to_load):
            st.error(f"❌ Direktori model tidak ditemukan di: {path_to_load}")
            return None

        st.info("🚀 Memuat model (SavedModel)...")
        model = tf.saved_model.load(path_to_load)
        st.success("✅ Model berhasil dimuat!")
        return model

    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_saved_model()
class_names = ['healthy', 'redrot', 'rust', 'yellow']

st.title("🌿 Sugarcane Leaf Disease Classifier")
st.markdown("Mode: **TensorFlow SavedModel**")

uploaded_file = st.file_uploader("📤 Upload gambar daun tebu", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼️ Gambar yang diupload", use_column_width=True)

    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    predict_fn = model.signatures["serving_default"]
    input_name = list(predict_fn.structured_input_signature[1].keys())[0]
    predictions = predict_fn(**{input_name: tf.convert_to_tensor(img_batch, dtype=tf.float32)})
    output_name = list(predict_fn.structured_outputs.keys())[0]
    predictions = predictions[output_name].numpy()
    
    pred_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    predicted_label = class_names[pred_index]

    # --- BAGIAN OUTPUT YANG DIPERBARUI ---
    st.markdown("### 🧠 Hasil Prediksi:")
    st.success(f"Kelas Prediksi: **{predicted_label}**")
    st.info(f"Keyakinan: **{confidence:.2f}%**")

    # Tambahan untuk menampilkan probabilitas semua kelas
    st.markdown("### 📊 Rincian Probabilitas:")
    for i, class_name in enumerate(class_names):
        probability = predictions[0][i] * 100
        st.write(f"- {class_name}: {probability:.2f}%")