from tensorflow.keras.models import load_model
import os

# Tentukan path ke folder model
model_folder_path = 'model\trained_model\best_model.keras'

# Cek apakah folder tersebut ada
if not os.path.isdir(model_folder_path):
    print(f"Error: Folder tidak ditemukan di '{model_folder_path}'")
else:
    try:
        # Muat model dari format folder (SavedModel)
        print(f"Mencoba memuat model dari folder '{model_folder_path}'...")
        model = load_model(model_folder_path)
        print("Model berhasil dimuat dari folder.")

        # Simpan kembali model dalam format .h5 tunggal
        output_path = 'model/converted_model.h5'
        print(f"Menyimpan model ke '{output_path}'...")
        model.save(output_path)

        print(f"Model berhasil dikonversi dan disimpan sebagai '{output_path}'")

    except Exception as e:
        print(f"Terjadi error saat memuat atau mengonversi model: {e}")