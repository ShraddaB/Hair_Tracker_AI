import numpy as np
import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# ---- Load your model ----
model_path = "hair_type_classifier.h5"
model = load_model(model_path)

# ---- Automatically pick one image from 'Curly Hair' folder ----
folder_path = r"C:\Users\asus\Desktop\Hair_Tracker_AI-main\Hair_Tracker_AI\datasets\Curly Hair"
img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not img_files:
    print("❌ No image files found in:", folder_path)
    exit()

# Pick a random image
img_name = random.choice(img_files)
img_path = os.path.join(folder_path, img_name)

print(f"\n🖼️ Testing image: {img_name}")

# ---- Load and preprocess the image ----
img = Image.open(img_path).convert("RGB")
img = img.resize((128, 128))

img_array = np.array(img, dtype="float32")
img_array = np.expand_dims(img_array, axis=0)

# ---- Try both preprocessing methods ----
img_div = img_array / 255.0
img_mobile = preprocess_input(img_array.copy())

pred1 = model.predict(img_div)[0]
pred2 = model.predict(img_mobile)[0]

print("\n=== Normal /255 Preprocessing ===")
print("Raw:", pred1)
print("Predicted index:", np.argmax(pred1))

print("\n=== MobileNet Preprocessing ===")
print("Raw:", pred2)
print("Predicted index:", np.argmax(pred2))
