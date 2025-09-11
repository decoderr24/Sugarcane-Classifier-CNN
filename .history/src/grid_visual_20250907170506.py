import matplotlib.pyplot as plt
import numpy as np

# Ambil 1 batch data dari generator
images, labels = next(train_generator)

# Ambil 1 gambar original (skala balik ke 0–255 biar tampil normal)
original = (images[0] * 255).astype("uint8")

# Buat list untuk hasil augmentasi
augmented = []
for aug_img, _ in zip(train_datagen.flow(np.expand_dims(images[0], 0), batch_size=1), range(6)):
    augmented.append((aug_img[0] * 255).astype("uint8"))

# Judul tiap tahapan augmentasi
titles = ["Original", "Brightness", "Rotation", "Width Shift", "Height Shift", "Shear", "Zoom/Flip"]

# =============================
# 1. Grid 1x7 (panjang horisontal)
# =============================
plt.figure(figsize=(18, 4))

plt.subplot(1, 7, 1)
plt.imshow(original)
plt.title(titles[0], fontsize=8)
plt.axis("off")

for i, img in enumerate(augmented, start=1):
    plt.subplot(1, 7, i+1)
    plt.imshow(img)
    plt.title(titles[i], fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.savefig("result/augmentation_grid_1x7.png", dpi=300)
plt.show()

# =============================
# 2. Grid 2x4 (lebih rapi untuk jurnal)
# =============================
plt.figure(figsize=(12, 6))

# Original + 6 augmentasi
all_images = [original] + augmented

for idx, img in enumerate(all_images):
    plt.subplot(2, 4, idx+1)
    plt.imshow(img)
    plt.title(titles[idx], fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig("result/augmentation_grid_2x4.png", dpi=300)
plt.show()
