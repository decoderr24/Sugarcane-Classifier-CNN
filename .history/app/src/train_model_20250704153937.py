import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint 
import seaborn as sns

# =================================================================================
# LANGKAH 1 & 2: PERSIAPAN DATA DAN AUGMENTASI
# =================================================================================

# Augmentasi untuk data training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,       
    width_shift_range=0.4,   
    height_shift_range=0.4,  
    shear_range=0.4,         
    zoom_range=0.4,          
    horizontal_flip=True,
    vertical_flip=True,      
    brightness_range=[0.7, 1.3],  # Variasi brightness
    channel_shift_range=0.2, 
    fill_mode='nearest'
)

# Untuk data validasi hanya rescale
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Definisikan batch size
batch_size = 8

# Path yang sudah diperbaiki (tanpa '/' di awal)
train_generator = train_datagen.flow_from_directory(
    'dataset/datatebu/train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    'dataset/datatebu/validation',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# =================================================================================
# LANGKAH 3: BANGUN ARSITEKTUR MODEL (DENGAN REGULARISASI LEBIH KUAT)
# =================================================================================

# Muat model dasar MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Bekukan (freeze) semua layer di model dasar
base_model.trainable = False

from tensorflow.keras.regularizers import l2
# Arsitektur dengan regularisasi lebih kuat
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.7),  # Tingkatkan dari 0.5
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.6),  # Tambah dropout layer
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')
])
# LANGKAH 4: KOMPILASI MODEL

# Mengonfigurasi proses belajar model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
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
initial_epochs = 30
fine_tune_epochs = 20
total_epochs = initial_epochs + fine_tune_epochs

# Siapkan callback untuk mengurangi learning rate secara otomatis
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    patience=2, 
    verbose=1, 
    factor=0.1, 
    min_lr=0.00001
)

# Tambahkan Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1,
    mode='min'
)

# Tambah ModelCheckpoint untuk menyimpan model terbaik
checkpoint = ModelCheckpoint(
    'model/best_model.keras',  # Simpan di folder model
    monitor='val_loss',
    save_best_only=True,
    verbose=1,
    mode='min'
)

# Gabungkan semua callbacks
callbacks = [learning_rate_reduction, early_stopping, checkpoint]

# --- Fase 1: Training Awal (Hanya layer atas) ---
print("\n--- Fase 1: Training Awal ---")
history = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=validation_generator,
    callbacks=callbacks  # Gunakan callbacks yang sudah lengkap
)

# --- Fase 2: Fine-Tuning ---
print("\n--- Fase 2: Memulai Fine-Tuning ---")

# Cairkan (unfreeze) base_model
base_model.trainable = True

# Bekukan semua layer kecuali layer teratas
fine_tune_at = 120
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Kompilasi ulang model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Lanjutkan training (fine-tuning)
history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=initial_epochs,
    validation_data=validation_generator,
    callbacks=callbacks  # Gunakan callbacks yang sama
)

