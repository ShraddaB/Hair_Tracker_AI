import streamlit as st
import os
from PIL import Image
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import os
print("✅ Running from:", os.getcwd())
print("✅ File path:", __file__)

# =======================
# Paths for models
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "AI_Models")

# =======================
# Class Labels
# =======================
HAIR_DISEASE_CLASSES = [
    "Alopecia Areata", "Contact Dermatitis", "Folliculitis", 
    "Head Lice", "Lichen Planus", "Male Pattern Baldness",
    "Psoriasis", "Seborrheic Dermatitis", "Telogen Effluvium", "Tinea Capitis"
]

# Hardcoded Hair Type classes for your dataset
HAIR_TYPE_CLASSES = ["Curly", "Straight", "Wavy"]

# =======================
# Load Models
# =======================
# Hairfall model
xgb_model_path = os.path.join(AI_MODELS_DIR, "Hairfall_Prediction", "XGB.joblib")
try:
    hairfall_model = joblib.load(xgb_model_path)
    hairfall_loaded = True
except Exception as e:
    hairfall_model = None
    hairfall_loaded = False
    st.warning(f"Hairfall model could not be loaded: {e}")

# Hair Disease model
disease_model_path = os.path.join(AI_MODELS_DIR, "Disease_Classification", "hair_disease_cnn_model.h5")
try:
    hair_disease_model = keras.models.load_model(disease_model_path)
    hair_disease_loaded = True
except Exception as e:
    hair_disease_model = None
    hair_disease_loaded = False
    st.warning(f"Hair disease model could not be loaded: {e}")

# Hair Type model
hair_type_model_path = os.path.join(AI_MODELS_DIR, "Type_Classification", "hair_type_classifier.h5")
try:
    hair_type_model = keras.models.load_model(hair_type_model_path)
    hair_type_loaded = True
except Exception as e:
    hair_type_model = None
    hair_type_loaded = False
    st.warning(f"Hair type model could not be loaded: {e}")

# =======================
# Sidebar Navigation
# =======================
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Hair Disease & Hair Type", "Hairfall Prediction"])

# =======================
# Hairfall Prediction Page
# =======================
if option == "Hairfall Prediction":
    st.title("Hairfall Prediction")

    if hairfall_loaded:
        pressure_level = st.selectbox("Pressure Level", [0, 1, 2, 3])
        stress_level = st.selectbox("Stress Level", [0, 1, 2, 3])
        hair_grease = st.selectbox("Hair Grease Level", [1.0, 2.0, 3.0, 4.0, 5.0])
        dandruff = st.selectbox("Dandruff Level", [1.0, 2.0])

        if st.button("Predict"):
            inputs = [[pressure_level, stress_level, hair_grease, dandruff]]
            prediction = hairfall_model.predict(inputs)[0]
            message = "might experience hairfall." if prediction == 1 else "might not experience hairfall."
            st.success(f"Prediction: The person {message}")
    else:
        st.warning("Hairfall model is missing or could not be loaded.")

# =======================
# Hair Disease & Hair Type Page
# =======================
if option == "Hair Disease & Hair Type":
    st.title("Hair Disease & Hair Type Prediction")

    # --- Hair Disease ---
    st.subheader("Upload the image of area affected by hair disease")
    disease_img_file = st.file_uploader("Choose a jpg/jpeg file", type=["jpg", "jpeg"], key="disease_img")

    if disease_img_file and hair_disease_loaded:
        img = Image.open(disease_img_file).convert("RGB")
        model_input_shape = hair_disease_model.input_shape
        if len(model_input_shape) == 4:
            target_size = (model_input_shape[1], model_input_shape[2])
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        else:
            img_array = np.array(img) / 255.0
            img_array = img_array.flatten()
            img_array = np.expand_dims(img_array, axis=0)

        prediction_idx = np.argmax(hair_disease_model.predict(img_array), axis=1)[0]
        st.image(img, caption="Uploaded Image")
        st.success(f"Predicted Disease Class: {HAIR_DISEASE_CLASSES[prediction_idx]}")
    elif disease_img_file:
        st.warning("Hair disease model is not loaded.")

    # --- Hair Type ---
    st.subheader("Upload an image of your hair to get hair type")
    hair_img_file = st.file_uploader("Choose a jpg/jpeg file", type=["jpg", "jpeg"], key="hair_img")

    def preprocess_image(img, target_size=(224, 224)):
        """Preprocess image: preserve aspect ratio, pad, normalize"""
        img = img.convert("RGB")
        img.thumbnail(target_size)
        new_img = Image.new("RGB", target_size, (0, 0, 0))
        new_img.paste(img, ((target_size[0]-img.size[0])//2, (target_size[1]-img.size[1])//2))
        img_array = np.array(new_img)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    if hair_img_file and hair_type_loaded:
        img = Image.open(hair_img_file)
        model_input_shape = hair_type_model.input_shape
        if len(model_input_shape) == 4:
            target_size = (model_input_shape[1], model_input_shape[2])
            img_array = preprocess_image(img, target_size)
        else:
            img_array = np.array(img.convert("RGB"))/255.0
            img_array = img_array.flatten()
            img_array = np.expand_dims(img_array, axis=0)

        prediction_idx = np.argmax(hair_type_model.predict(img_array), axis=1)[0]
        st.image(img, caption="Uploaded Hair Image")
        st.success(f"Predicted Hair Type Class: {HAIR_TYPE_CLASSES[prediction_idx]}")
    elif hair_img_file:
        st.warning("Hair type model is not loaded.")

    # --- Hair Conditions and Treatments Table ---
    st.markdown("## Hair Conditions and Treatments")
    hair_disease_table = pd.DataFrame({
        "Disease Name": ["Alopecia Areata", "Contact Dermatitis", "Folliculitis", "Head Lice", "Lichen Planus",
                         "Male Pattern Baldness", "Psoriasis", "Seborrheic Dermatitis", "Telogen Effluvium", "Tinea Capitis"],
        "Treatment 1": ["Corticosteroids", "Topical corticosteroids", "Warm compresses", "OTC lice meds", "Topical corticosteroids",
                        "Minoxidil", "Topical corticosteroids", "Topical antifungal", "Observation", "Oral antifungal meds"],
        "Treatment 2": ["Minoxidil", "Antihistamines", "Topical antibiotics", "Fine-toothed comb", "Antimalarial drugs",
                        "Finasteride", "Light therapy", "Topical corticosteroids", "Iron supplements", "Topical antifungal shampoo"],
        "Treatment 3": ["Immunosuppressants", "Avoidance of irritants", "Oral antibiotics", "Wash bedding & clothing", "Systemic corticosteroids",
                        "Hair transplant", "Systemic medications", "Shampoo with selenium sulfide", "Stress management", "Selenium sulfide shampoo"]
    })
    st.dataframe(hair_disease_table)
