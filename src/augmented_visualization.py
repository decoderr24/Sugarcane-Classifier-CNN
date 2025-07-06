import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# =================================================================================
# 1. KONFIGURASI
# =================================================================================

# Definisikan path ke data training dan folder output utama
TRAIN_DIR = 'dataset/datatebu/train'
OUTPUT_DIR = 'result/augmented_visualization'
# Pilih satu kelas saja untuk dijadikan contoh visualisasi
CLASS_TO_VISUALIZE = 'healthy' 
NUM_EXAMPLES_TO_GENERATE = 5 # Jumlah contoh yang akan dibuat per augmentasi

# Pastikan folder output utama ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =================================================================================
# 2. DEFINISIKAN SETIAP AUGMENTASI SECARA TERPISAH
# =================================================================================

# Buat dictionary yang berisi nama augmentasi dan 'resep' generatornya
augmentation_generators = {
    'rotation': ImageDataGenerator(rotation_range=25, fill_mode='nearest'),
    'zoom': ImageDataGenerator(zoom_range=[0.7, 1.3], fill_mode='nearest'), # Zoom in dan out
    'width_shift': ImageDataGenerator(width_shift_range=0.3, fill_mode='nearest'),
    'height_shift': ImageDataGenerator(height_shift_range=0.3, fill_mode='nearest'),
    'shear': ImageDataGenerator(shear_range=0.2, fill_mode='nearest'),
    'horizontal_flip': ImageDataGenerator(horizontal_flip=True),
    'brightness': ImageDataGenerator(brightness_range=[0.6, 1.4])
}

# =================================================================================
# 3. PROSES GENERASI DAN PENYIMPANAN GAMBAR
# =================================================================================

print(f"Memproses sampel dari kelas '{CLASS_TO_VISUALIZE}' untuk visualisasi augmentasi...")

# Ambil satu gambar contoh dari kelas yang dipilih
try:
    class_path = os.path.join(TRAIN_DIR, CLASS_TO_VISUALIZE)
    sample_image_name = os.listdir(class_path)[0] # Ambil gambar pertama sebagai sampel
    sample_image_path = os.path.join(class_path, sample_image_name)
    
    print(f"  - Menggunakan gambar sampel: {sample_image_name}")

    # Muat gambar sampel
    img = load_img(sample_image_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape) # Reshape agar bisa diproses

    # Loop melalui setiap jenis augmentasi yang telah kita definisikan
    for aug_name, datagen in augmentation_generators.items():
        print(f"  - Menghasilkan augmentasi untuk: '{aug_name}'...")
        
        # Buat folder output spesifik untuk jenis augmentasi ini
        output_aug_path = os.path.join(OUTPUT_DIR, aug_name)
        os.makedirs(output_aug_path, exist_ok=True)
        
        # Hasilkan dan simpan beberapa contoh
        i = 0
        for batch in datagen.flow(x, 
                                  batch_size=1,
                                  save_to_dir=output_aug_path, 
                                  save_prefix=f'{aug_name}_example', 
                                  save_format='jpeg'):
            i += 1
            if i >= NUM_EXAMPLES_TO_GENERATE:
                break
                
except Exception as e:
    print(f"Tidak dapat memproses. Pastikan folder '{class_path}' tidak kosong. Error: {e}")

print("\nProses selesai.")
print(f"Gambar visualisasi augmentasi telah disimpan di folder: '{OUTPUT_DIR}'")
