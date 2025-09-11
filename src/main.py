import os
import streamlit as st
import tensorflow as tf
import gdown
import numpy as np
from PIL import Image

st.set_page_config(page_title="Sugarcane Classifier", layout="centered")

# ===============================
# Load model (PATH SUDAH SESUAI)
# ===============================
@st.cache_resource
def load_trained_model():
    """Load model from Google Drive if not exists locally"""
    try:
        # Google Drive folder/file ID containing the model
        model_url = "YOUR_GOOGLE_DRIVE_ID"
        model_path = os.path.join(os.path.dirname(__file__), "model")
        
        if not os.path.exists(model_path):
            st.info("⏳ Downloading model files...")
            gdown.download_folder(url=model_url, output=model_path, quiet=False)
            
        model = tf.saved_model.load(os.path.join(model_path, "best_model_export"))
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
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

    # Preprocessing Gambar
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Prediksi
    predict_fn = model.signatures["serving_default"]
    
    # Get the input tensor name from model signature
    input_name = list(predict_fn.structured_input_signature[1].keys())[0]
    
    # Make prediction with correct input name
    predictions = predict_fn(**{input_name: tf.convert_to_tensor(img_batch, dtype=tf.float32)})
    
    # Get the output tensor name
    output_name = list(predict_fn.structured_outputs.keys())[0]
    predictions = predictions[output_name].numpy()
    
    pred_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
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