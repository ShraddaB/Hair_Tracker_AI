import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ---- Correct model path ----
model_path = r"C:\Users\asus\Desktop\Hair_Tracker_AI-main\AI_Models\hair_type_classifier.h5"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at: {model_path}")

model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# ---- Test image ----
img_path = r"C:\Users\asus\Desktop\Hair_Tracker_AI-main\Hair_Tracker_AI\datasets\Curly Hair\02dac897d1dec9ba8c057a11d041ada8--layered-natural-hair-natural-black-hairstyles.jpg"

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found at: {img_path}")

# ---- Load and preprocess image ----

img = image.load_img(img_path, target_size=(128,128))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# ---- Predict ----
pred = model.predict(img_array)
classes = ["Curly", "Straight", "Wavy"]
predicted_class = classes[np.argmax(pred)]

print("\n=== Prediction Result ===")
print("Raw prediction values:", pred[0])
print(f"Predicted class: {predicted_class}")
