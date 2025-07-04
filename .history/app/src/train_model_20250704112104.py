import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# =================================================================================
# LANGKAH 1 & 2: PERSIAPAN DATA DAN AUGMENTASI
# =================================================================================

# Augmentasi untuk data training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Untuk data validasi hanya rescale
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Path yang sudah diperbaiki (tanpa '/' di awal)
train_generator = train_datagen.flow_from_directory(
    'dataset/datatebu/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    'dataset/datatebu/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False # Penting: jangan acak data validasi untuk confusion matrix
)

# =================================================================================
# LANGKAH 3: BANGUN ARSITEKTUR MODEL (TRANSFER LEARNING)
# =================================================================================

# Muat model dasar MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Bekukan (freeze) semua layer di model dasar
base_model.trainable = False

# Tumpuk layer kustom Anda di atas model dasar
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='softmax')
])

# =================================================================================
# LANGKAH 4: KOMPILASI MODEL
# =================================================================================

# Mengonfigurasi proses belajar model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Menampilkan ringkasan arsitektur model
print("\n--- Ringkasan Arsitektur Model ---")
model.summary()

# =================================================================================
# LANGKAH 5: LATIH MODEL
# =================================================================================
print("\nMemulai proses training model...")
EPOCHS = 20

# Memulai proses training dengan data generator
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

print("Proses training selesai.")

# =================================================================================
# LANGKAH 6: EVALUASI & SIMPAN HASIL
# =================================================================================

# Ambil catatan akurasi dan loss dari variabel 'history'
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Tentukan rentang epoch
epochs_range = range(EPOCHS)

# Buat plot untuk Akurasi dan Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Simpan gambar plot ke folder result
plot_path = os.path.join('result', 'training_history_plot.png')
plt.savefig(plot_path)
print(f"\nPlot riwayat training disimpan di: {plot_path}")
plt.show()

# >>> BAGIAN BARU UNTUK CONFUSION MATRIX <<<
print("\nMembuat Confusion Matrix untuk data validasi...")

# Mendapatkan prediksi model pada data validasi
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Mendapatkan label yang sebenarnya
y_true = validation_generator.classes

# Menghitung confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Menyiapkan label untuk plot
class_names = list(validation_generator.class_indices.keys())

# Membuat plot confusion matrix menggunakan Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d', # Format angka sebagai integer
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title('Confusion Matrix')
plt.ylabel('Kelas Sebenarnya (Actual)')
plt.xlabel('Kelas Prediksi (Predicted)')

# Simpan gambar plot ke folder result
cm_plot_path = os.path.join('result', 'confusion_matrix.png')
plt.savefig(cm_plot_path)
print(f"Confusion Matrix disimpan di: {cm_plot_path}")
plt.show()
# >>> AKHIR DARI BAGIAN BARU <<<

# Simpan model yang sudah dilatih ke dalam satu file
model_path = os.path.join('model', 'sugarcane_classifier_model.keras')
model.save(model_path)
print(f"Model berhasil disimpan di: {model_path}")