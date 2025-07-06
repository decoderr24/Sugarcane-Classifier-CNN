import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# =================================================================================
# 1. KONFIGURASI
# =================================================================================

# Definisikan path ke data training dan folder output
TRAIN_DIR = 'dataset/datatebu/train'
OUTPUT_DIR = 'result/augmented_examples'
CLASSES = ['healthy', 'redrot', 'rust', 'yellow']
NUM_AUGMENTATIONS_PER_IMAGE = 5 # Jumlah variasi augmentasi yang akan dibuat per gambar

# Pastikan folder output ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =================================================================================
# 2. DEFINISIKAN "RESEP" AUGMENTASI
# =================================================================================
# Gunakan parameter augmentasi yang sama persis dengan script training Anda
augmentation_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    fill_mode='nearest'
)

# =================================================================================
# 3. PROSES GENERASI DAN PENYIMPANAN GAMBAR
# =================================================================================

print("Memulai proses generasi gambar hasil augmentasi...")

# Loop melalui setiap kelas
for class_name in CLASSES:
    class_path = os.path.join(TRAIN_DIR, class_name)
    output_class_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_path, exist_ok=True)
    
    # Ambil satu gambar contoh dari setiap kelas
    try:
        sample_image_name = os.listdir(class_path)[0]
        sample_image_path = os.path.join(class_path, sample_image_name)
        
        print(f"  - Memproses sampel dari kelas '{class_name}': {sample_image_name}")

        # Muat gambar
        img = load_img(sample_image_path)
        # Ubah gambar menjadi array numpy
        x = img_to_array(img)
        # Reshape menjadi (1, height, width, channels) untuk bisa diproses generator
        x = x.reshape((1,) + x.shape)

        # Gunakan .flow() untuk menghasilkan gambar augmentasi dan menyimpannya
        # Ini adalah cara paling efisien untuk menyimpan hasil augmentasi
        i = 0
        for batch in augmentation_datagen.flow(x, 
                                             batch_size=1,
                                             save_to_dir=output_class_path, 
                                             save_prefix=f'aug_{class_name}', 
                                             save_format='jpeg'):
            i += 1
            if i >= NUM_AUGMENTATIONS_PER_IMAGE:
                break  # Hentikan setelah membuat 5 gambar
                
    except Exception as e:
        print(f"Tidak dapat memproses kelas '{class_name}'. Pastikan folder tidak kosong. Error: {e}")

print("\nProses selesai.")
print(f"Gambar hasil augmentasi telah disimpan di folder: '{OUTPUT_DIR}'")

