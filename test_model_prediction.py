import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import os

# ---- Load your model ----
model_path = os.path.join("AI_Models", "Type_Classification", "hair_type_classifier.h5")
model = load_model(model_path)

# ---- Load a known curly hair image ----
img_path = "your_curly_image.jpg"  # 👉 replace with your curly image filename
img = Image.open(img_path).convert("RGB")
img = img.resize((128, 128))

img_array = np.array(img, dtype="float32")
img_array = np.expand_dims(img_array, axis=0)

# ---- Try both normal and MobileNet preprocessing ----
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
