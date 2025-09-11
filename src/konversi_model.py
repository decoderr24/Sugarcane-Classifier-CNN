from tensorflow import keras

model = keras.models.load_model("model/sugarcane_classifier_model.keras")
model.save("model/best_model_v2.keras")

print("✅ Model berhasil disimpan ulang ke best_model_v2.keras")
