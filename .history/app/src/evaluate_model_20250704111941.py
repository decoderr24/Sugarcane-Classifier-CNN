import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# =================================================================================
# 1. DEFINISIKAN PATH DAN PARAMETER
# =================================================================================
MODEL_PATH = os.path.join('model', 'sugarcane_classifier_model.keras')
VALIDATION_DIR = os.path.join('dataset', 'datatebu', 'validation')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =================================================================================
# 2. MUAT MODEL YANG SUDAH DILATIH
# =================================================================================
print(f"Memuat model dari: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# =================================================================================
# 3. SIAPKAN GENERATOR DATA VALIDASI
# =================================================================================
# Untuk data validasi hanya perlu rescale, tidak perlu augmentasi
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Sangat PENTING: jangan acak data untuk evaluasi
)

# =================================================================================
# 4. LAKUKAN PREDIKSI DAN BUAT CONFUSION MATRIX
# =================================================================================
print("\nMelakukan prediksi pada data validasi untuk membuat Confusion Matrix...")

# Mendapatkan prediksi model
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Mendapatkan label yang sebenarnya
y_true = validation_generator.classes

# Menghitung confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Menyiapkan label kelas untuk plot
class_names = list(validation_generator.class_indices.keys())

# Membuat plot confusion matrix menggunakan Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',  # Format angka sebagai integer
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title('Confusion Matrix - Validation Data')
plt.ylabel('Kelas Sebenarnya (Actual)')
plt.xlabel('Kelas Prediksi (Predicted)')

# Simpan gambar plot ke folder result
cm_plot_path = os.path.join('result', 'confusion_matrix_evaluation.png')
plt.savefig(cm_plot_path)
print(f"Confusion Matrix disimpan di: {cm_plot_path}")

# Tampilkan plot
plt.show()