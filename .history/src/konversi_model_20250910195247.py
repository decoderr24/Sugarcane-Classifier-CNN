from tensorflow.keras.models import load_model

# Muat model .keras Anda yang sudah ada
model = load_model('model/best_model.keras')

# Simpan kembali model dalam format .h5
model.save('model/best_model.h5')

print("Model berhasil dikonversi ke best_model.h5")