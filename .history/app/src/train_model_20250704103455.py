import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Augmentasi untuk data training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,              # Wajib: Mengubah skala piksel menjadi 0-1
    rotation_range=20,           # Mengacak rotasi gambar hingga 20 derajat
    width_shift_range=0.2,       # Mengacak pergeseran horizontal
    height_shift_range=0.2,      # Mengacak pergeseran vertikal
    shear_range=0.2,             # Mengacak pemotongan sudut
    zoom_range=0.2,              # Mengacak zoom
    horizontal_flip=True,        # Mengizinkan gambar dibalik horizontal
    fill_mode='nearest'          # Mengisi piksel yang mungkin kosong setelah transformasi
)

# Untuk data validasi hanya rescale
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    '../dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    '../dataset/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# =================================================================================
# LANGKAH 3: BANGUN ARSITEKTUR MODEL (TRANSFER LEARNING)
# =================================================================================

# Muat model dasar MobileNetV2 yang sudah dilatih di dataset ImageNet
# include_top=False berarti kita tidak menyertakan layer klasifikasi teratasnya
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Bekukan (freeze) semua layer di model dasar agar pengetahuannya tidak rusak.
base_model.trainable = False

# Tumpuk layer kustom Anda di atas model dasar
model = tf.keras.models.Sequential([
    # 1. Fondasi model untuk ekstraksi fitur
    base_model,
    
    # 2. Layer untuk meratakan output dari base_model
    tf.keras.layers.GlobalAveragePooling2D(),
    
    # 3. Layer Dropout untuk mencegah overfitting
    tf.keras.layers.Dropout(0.3),
    
    # 4. Layer output dengan 4 neuron (untuk 4 kelas) dan aktivasi softmax
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

# KOMPILASI MODEL
print("\n--- Kompilasi Model ---")
model.summary()
# Mengonfigurasi proses belajar model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Menampilkan ringkasan arsitektur model yang sudah kita bangun
print("\n--- Ringkasan Arsitektur Model ---")
model.summary()

# LANGKAH 5: LATIH MODEL
print("\nMemulai proses training model...")
EPOCHS = 20

# Memulai proses training dengan data generator
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

print("Proses training selesai.")