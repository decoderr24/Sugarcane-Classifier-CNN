import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load satu gambar contoh (ganti dengan path dataset kamu)
img_path = "dataset/datatebu/test/rust/DSCN0837.JPG"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224,224))  # Sesuai input MobileNetV2
img = np.expand_dims(img, axis=0) / 255.0

# Augmentasi generator (sesuai setting penelitian kamu)
datagen = ImageDataGenerator(
    brightness_range=[0.6, 1.4],
    rotation_range=25,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True
)

# Hasilkan beberapa augmentasi
aug_iter = datagen.flow(img, batch_size=1)

# Simpan 1 original + 6 hasil augmentasi
images = [img[0]]
for i in range(6):  
    batch = next(aug_iter)
    images.append(batch[0])

# Plot dalam grid (1x7)
plt.figure(figsize=(1, 4))
titles = ["Original", "Brightness", "Rotation", "Width Shift", "Height Shift", "Shear", "Zoom/Flip"]

for i in range(8):  # misalnya 1 original + 7 augmentasi
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i])
    plt.title(titles[i], fontsize=8)
    plt.axis("off")
plt.tight_layout()
plt.savefig("result/augmentation_grid_2x4.png", dpi=300)
plt.show()
