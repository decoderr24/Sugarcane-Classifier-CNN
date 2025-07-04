import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# =================================================================================
# 1. DEFINISIKAN PATH DAN PARAMETER
# =================================================================================
MODEL_PATH = os.path.join('model', 'best_model.keras') # Menggunakan model terbaik yang disimpan
TEST_DIR = os.path.join('dataset', 'datatebu', 'test') # Mengarah ke folder test
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =================================================================================
# 2. MUAT MODEL YANG SUDAH DILATIH
# =================================================================================
print(f"Memuat model dari: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# =================================================================================
# 3. SIAPKAN GENERATOR DATA TEST
# =================================================================================
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# =================================================================================
# 4. LAKUKAN PREDIKSI DAN BUAT CONFUSION MATRIX
# =================================================================================
print("\nMelakukan prediksi pada data test...")

# Prediksi model
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

# Cetak Laporan Klasifikasi (Precision, Recall, F1-Score)
print("\n--- Laporan Klasifikasi (Test Data) ---")
print(classification_report(y_true, y_pred, target_names=class_names))

# Buat dan visualisasikan Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title('Confusion Matrix - Test Data')
plt.ylabel('Kelas Sebenarnya (Actual)')
plt.xlabel('Kelas Prediksi (Predicted)')

plt.show()