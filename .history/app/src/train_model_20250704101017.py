import tensorflow as tf
# Membuat "resep" augmentasi untuk data training
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

# Untuk data validasi, kita tidak melakukan augmentasi, hanya rescale
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)