import tensorflow as tf
from tensorflow import keras
import os

# Base directory where models are located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# List of model paths to convert
models_to_convert = [
    os.path.join(BASE_DIR, "Disease_Classification", "hair_disease_cnn_model.h5"),
    os.path.join(BASE_DIR, "Hair_Type", "hair_type_cnn_model.h5")
]

for model_path in models_to_convert:
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found: {model_path}")
        continue

    print(f"🔄 Converting: {model_path}")
    try:
        model = keras.models.load_model(model_path, compile=False)
        new_model_path = model_path.replace(".h5", "_converted.h5")
        model.save(new_model_path, save_format="h5")
        print(f"✅ Saved converted model at: {new_model_path}\n")
    except Exception as e:
        print(f"❌ Failed to convert {model_path}: {e}\n")
