import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Sugarcane Classifier", layout="centered")

# ===============================
# Load model (PATH SUDAH SESUAI)
# ===============================
@st.cache_resource
def load_trained_model():
    """Memuat model Keras terbaik yang sudah dilatih."""
    model_path = os.path.join("model", "best_model_fixed.h5")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error saat memuat model dari path '{model_path}': {e}")
        return None

model = load_trained_model()

# ===============================
# Daftar nama kelas (ubah sesuai dataset)
# ===============================
class_names = ['healthy', 'redrot', 'rust', 'yellow']

# ===============================
# Tampilan Streamlit
# ===============================
st.title("🌿 Sugarcane Leaf Disease Classifier")
st.markdown(
    "Upload gambar daun tebu dan sistem akan memprediksi jenis penyakitnya "
    "menggunakan model CNN MobileNetV2."
)

uploaded_file = st.file_uploader("📤 Upload gambar daun tebu", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼️ Gambar yang diupload", use_column_width=True)

    # ===============================
    # Preprocessing Gambar
    # ===============================
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # ===============================
    # Prediksi
    # ===============================
    predictions = model.predict(img_batch)
    pred_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    predicted_label = class_names[pred_index]

    # ===============================
    # Output
    # ===============================
    st.markdown("### 🧠 Hasil Prediksi:")
    st.success(f"Kelas Prediksi: **{predicted_label}**")
    st.info(f"Akurasi : **{confidence:.2f}%**")

    st.markdown("### 📊 Probabilitas Semua Kelas:")
    for i, name in enumerate(class_names):
        st.write(f"- {name}: {predictions[0][i]*100:.2f}%")
