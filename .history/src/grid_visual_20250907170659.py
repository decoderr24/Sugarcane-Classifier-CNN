import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Sama dengan yang ada di train.py
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
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

train_generator = train_datagen.flow_from_directory(
    'dataset/datatebu/test/rust/DSCN0837.JPG',
    target_size=(224, 224),
    batch_size=1,     # cukup 1 gambar saja
    class_mode='categorical'
)

# Ambil 1 gambar original
images, labels = next(train_generator)
original = (images[0] * 255).astype("uint8")

# Buat list untuk hasil augmentasi
augmented = []
for aug_img, _ in zip(train_datagen.flow(np.expand_dims(images[0], 0), batch_size=1), range(7)):
    augmented.append((aug_img[0] * 255).astype("uint8"))

titles = ["Original", "Brightness", "Rotation", "Width Shift", "Height Shift", "Shear", "Zoom", "Horizontal Flip"]

# =============================
# Grid 2x4 (lebih cocok untuk jurnal)
# =============================
plt.figure(figsize=(12, 6))
all_images = [original] + augmented

for idx, img in enumerate(all_images):
    plt.subplot(2, 4, idx+1)
    plt.imshow(img.astype("uint8"))
    plt.title(titles[idx], fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig("result/augmentation_grid.png", dpi=300)
plt.show()
