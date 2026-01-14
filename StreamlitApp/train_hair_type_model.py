# ========================================
# Hair Type Classifier (Straight, Wavy, Curly)
# Using Transfer Learning - MobileNetV2
# ========================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# =========================
# ✅ Paths
# =========================
# Folder structure should look like:
# dataset/
# ├── train/
# │   ├── Straight/
# │   ├── Wavy/
# │   └── Curly/
# ├── val/
# │   ├── Straight/
# │   ├── Wavy/
# │   └── Curly/
# └── test/
#     ├── Straight/
#     ├── Wavy/
#     └── Curly/

base_dir = r"C:\Users\asus\Desktop\Hair_Tracker_AI-main\Hair_Tracker_AI\datasets"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")



# =========================
# ✅ Image Data Generators
# =========================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =========================
# ✅ Build Model
# =========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

base_model.trainable = False  # Freeze pretrained layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
predictions = Dense(3, activation="softmax")(x)  # 3 classes: Straight, Wavy, Curly

model = Model(inputs=base_model.input, outputs=predictions)

# =========================
# ✅ Compile Model
# =========================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# ✅ Train Model
# =========================
EPOCHS = 15

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# =========================
# ✅ Evaluate Model
# =========================
test_loss, test_acc = model.evaluate(test_gen)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

# =========================
# ✅ Save Model
# =========================
os.makedirs("saved_models", exist_ok=True)
model.save("saved_models/hair_type_classifier.h5")
print("\n🎉 Model saved successfully at 'saved_models/hair_type_classifier.h5'")

# =========================
# ✅ Optional: Unfreeze & Fine-tune
# =========================
# To boost accuracy >90%, unfreeze last few layers:
# base_model.trainable = True
# for layer in base_model.layers[:-30]:
#     layer.trainable = False
# model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
# model.fit(train_gen, validation_data=val_gen, epochs=5)
