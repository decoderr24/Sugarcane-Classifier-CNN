import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau
import seaborn as sns

# =================================================================================
# LANGKAH 1 & 2: PERSIAPAN DATA DAN AUGMENTASI
# =================================================================================

# Augmentasi untuk data training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
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
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')
])

# LANGKAH 4: KOMPILASI MODEL

# Mengonfigurasi proses belajar model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Menampilkan ringkasan arsitektur model
print("\n--- Ringkasan Arsitektur Model ---")
model.summary()

# Tambahkan impor ini di bagian atas file Anda

# LANGKAH 5: LATIH MODEL DENGAN FINE-TUNING
print("\nMemulai proses training model...")

# Tentukan jumlah epoch untuk setiap fase
initial_epochs = 20
fine_tune_epochs = 15
total_epochs = initial_epochs + fine_tune_epochs

# Siapkan callback untuk mengurangi learning rate secara otomatis
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy', 
    patience=2, 
    verbose=1, 
    factor=0.5, 
    min_lr=0.00001
)

# --- Fase 1: Training Awal (Hanya layer atas) ---
print("\n--- Fase 1: Training Awal ---")
history = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=validation_generator,
    callbacks=[learning_rate_reduction] # Tambahkan callback di sini
)

# --- Fase 2: Fine-Tuning ---
print("\n--- Fase 2: Memulai Fine-Tuning ---")

# Cairkan (unfreeze) base_model.
base_model.trainable = True

# Mari kita lihat ada berapa layer di base model
print("Jumlah layer di base model: ", len(base_model.layers))

# Bekukan semua layer kecuali 40 layer teratas
fine_tune_at = 120 # Anda bisa bereksperimen dengan angka ini
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Kompilasi ulang model dengan learning rate yang SANGAT KECIL
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), # Learning rate rendah untuk fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Lanjutkan training (fine-tuning)
history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=initial_epochs,
    validation_data=validation_generator,
    callbacks=[learning_rate_reduction] # Gunakan callback lagi
)

# Gabungkan history dari kedua fase training
history.history['accuracy'].extend(history_fine.history['accuracy'])
history.history['val_accuracy'].extend(history_fine.history['val_accuracy'])
history.history['loss'].extend(history_fine.history['loss'])
history.history['val_loss'].extend(history_fine.history['val_loss'])

print("Proses training dan fine-tuning selesai.")

# Di Langkah 6, ubah `epochs_range = range(EPOCHS)` menjadi:
# epochs_range = range(total_epochs)

# LANGKAH 6: EVALUASI & SIMPAN HASIL

# Ambil catatan akurasi dan loss dari variabel 'history'
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Tentukan rentang epoch
epochs_range = range(total_epochs)

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