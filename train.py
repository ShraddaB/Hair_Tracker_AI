import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# === PATH SETUP ===
dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


# === PARAMETERS ===
img_size = (128, 128)
batch_size = 32
epochs = 15

# === DATA GENERATORS ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === MODEL DEFINITION ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

# === COMPILE ===
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === TRAIN ===
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

# === SAVE MODEL ===
save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AI_Models")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "hair_type_classifier.h5")
model.save(save_path)

print(f"✅ Hair Type Model trained successfully and saved at: {save_path}")

# === PRINT CLASS ORDER ===
print("\n🧠 Class indices (VERY IMPORTANT):")
print(train_gen.class_indices)
print("👉 Use this order when mapping predictions in Streamlit.")
