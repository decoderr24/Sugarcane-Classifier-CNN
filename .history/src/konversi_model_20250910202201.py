from tensorflow.keras.models import load_model

# Muat model sugarcane_classifier_model.keras
print("Mencoba memuat model/sugarcane_classifier_model.keras...")
model/best_model.keras
print("Model berhasil dimuat.")

# Simpan kembali model dalam format .h5
print("Menyimpan model ke model/sugarcane_classifier_model.h5...")
model.save('model/sugarcane_classifier_model.h5')

print("Model berhasil dikonversi ke sugarcane_classifier_model.h5")