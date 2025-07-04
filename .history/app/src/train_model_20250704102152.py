import tensorflow as tf
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

# Lanjutan kode di src/train_model.py

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