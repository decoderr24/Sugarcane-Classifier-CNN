from tensorflow.keras.models import load_model

# Muat model .keras Anda yang sudah ada
model = load_model('model/sugarcane_classifier_model.keras')

# Simpan kembali model dalam format .h5
# Simpan kembali model dalam format .h5
model.save('model/sugarcane_classifier_model.h5') # Kita simpan dengan nama baru

print("Model berhasil dikonversi ke best_model.h5")