import matplotlib.pyplot as plt
import numpy as np

# Ambil 1 batch data dari generator
images, labels = next(train_generator)

# Ambil 1 gambar original
original = (images[0] * 255).astype("uint8")

# Buat list augmented
augmented = []
for aug_img, _ in zip(train_datagen.flow(np.expand_dims(images[0], 0), batch_size=1), range(6)):
    augmented.append((aug_img[0] * 255).astype("uint8"))

# Plot
plt.figure(figsize=(18, 4))
titles = ["Original", "Brightness", "Rotation", "Width Shift", "Height Shift", "Shear", "Zoom/Flip"]

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
plt.savefig("result/augmentation_grid.png", dpi=300)
plt.show()
