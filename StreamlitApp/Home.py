import streamlit as st
import os
from PIL import Image
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from tensorflow.keras.applications import mobilenet_v2, vgg16, resnet50
import base64


# =======================
# Set Background Image (behind content)
# =======================
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        /* App background */
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}

        /* Light overlay for readability */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.55);
            z-index: 0;
        }}

        /* Content stays above background */
        .block-container {{
            position: relative;
            z-index: 1;
        }}

        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: rgba(240, 248, 255, 0.9);
            color: #333333;
            font-family: 'Segoe UI', sans-serif;
        }}

        [data-testid="stSidebar"] h2 {{
            color: #4B0082;
            text-align: center;
            font-size: 1.8rem;
        }}

        .main-title {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #4B0082;
            text-align: center;
            text-shadow: 1px 1px 3px white;
        }}

        .subheader {{
            font-size: 1.5rem;
            color: #FF4500;
            text-shadow: 1px 1px 3px white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


image_path = r"C:\Users\asus\OneDrive\Desktop\Hair_Tracker_AI-main\Hair_Tracker_AI\StreamlitApp\hair1.png"
if os.path.exists(image_path):
    set_background(image_path)
else:
    st.warning("⚠️ Background image not found at the specified path.")


# =======================
# Model Paths
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
HAIR_TYPE_CLASSES = ["Straight", "Wavy", "Curly", "Coily"]

# =======================
# Load Models
# =======================
xgb_model_path = os.path.join(AI_MODELS_DIR, "Hairfall_Prediction", "XGB.joblib")
try:
    hairfall_model = joblib.load(xgb_model_path)
    hairfall_loaded = True
except Exception as e:
    hairfall_model = None
    hairfall_loaded = False
    st.warning(f"⚠️ Hairfall model could not be loaded: {e}")

disease_model_path = os.path.join(AI_MODELS_DIR, "Disease_Classification", "hair_disease_cnn_model.h5")
try:
    hair_disease_model = keras.models.load_model(disease_model_path)
    hair_disease_loaded = True
except Exception as e:
    hair_disease_model = None
    hair_disease_loaded = False
    st.warning(f"⚠️ Hair disease model could not be loaded: {e}")

hair_type_model_path = os.path.join(AI_MODELS_DIR, "Type_Classification", "hair_type_classifier.h5")
try:
    hair_type_model = keras.models.load_model(hair_type_model_path)
    hair_type_loaded = True
except Exception as e:
    hair_type_model = None
    hair_type_loaded = False
    st.warning(f"⚠️ Hair type model could not be loaded: {e}")

# =======================
# Sidebar Navigation
# =======================
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Hair Disease & Hair Type", "Hairfall Prediction"])

# =======================
# Hairfall Prediction Page
# =======================
if option == "Hairfall Prediction":
    st.markdown('<div class="main-title">💇 Hairfall Prediction</div>', unsafe_allow_html=True)

    if hairfall_loaded:
        pressure_level = st.selectbox("Pressure Level", [0, 1, 2, 3])
        stress_level = st.selectbox("Stress Level", [0, 1, 2, 3])
        hair_grease = st.selectbox("Hair Grease Level", [1.0, 2.0, 3.0, 4.0, 5.0])
        dandruff = st.selectbox("Dandruff Level", [1.0, 2.0])

        if st.button("Predict"):
            inputs = [[pressure_level, stress_level, hair_grease, dandruff]]
            prediction = hairfall_model.predict(inputs)[0]
            message = "might experience hairfall." if prediction == 1 else "might not experience hairfall."
            st.markdown(f'<div class="prediction-box">Prediction: The person {message}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Hairfall model is missing or could not be loaded.")

# =======================
# Hair Disease & Hair Type Page
# =======================
if option == "Hair Disease & Hair Type":
    st.markdown('<div class="main-title">🧠 Hair Disease & Hair Type Prediction</div>', unsafe_allow_html=True)

    st.markdown('<div class="subheader">🔬 Upload the image of area affected by hair disease</div>', unsafe_allow_html=True)
    disease_img_file = st.file_uploader("Choose a jpg/jpeg file", type=["jpg", "jpeg"], key="disease_img")

    if disease_img_file and hair_disease_loaded:
        img = Image.open(disease_img_file).convert("RGB")
        input_shape = hair_disease_model.input_shape
        target_size = (input_shape[1], input_shape[2]) if len(input_shape) >= 3 else (150, 150)
        img = img.resize(target_size)

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = hair_disease_model.predict(img_array)
        prediction_idx = np.argmax(prediction, axis=1)[0]

        st.image(img, caption="Uploaded Image", width=400)
        st.markdown(f'<div class="prediction-box">🧬 Predicted Disease Class: {HAIR_DISEASE_CLASSES[prediction_idx]}</div>', unsafe_allow_html=True)

    elif disease_img_file:
        st.warning("⚠️ Hair disease model is not loaded.")

    st.markdown('<div class="subheader">💁 Upload an image of your hair to get hair type</div>', unsafe_allow_html=True)
    hair_img_file = st.file_uploader("Choose a jpg/jpeg file", type=["jpg", "jpeg"], key="hair_img")

    if hair_img_file and hair_type_loaded:
        img = Image.open(hair_img_file).convert("RGB")
        input_shape = hair_type_model.input_shape
        target_size = (input_shape[1], input_shape[2]) if len(input_shape) >= 3 else (128, 128)
        img = img.resize(target_size)
        img_array = np.asarray(img).astype("float32")

        model_name = getattr(hair_type_model, "name", "").lower()
        try:
            if "mobilenet" in model_name:
                img_array = mobilenet_v2.preprocess_input(img_array)
            elif "vgg" in model_name:
                img_array = vgg16.preprocess_input(img_array)
            elif "resnet" in model_name:
                img_array = resnet50.preprocess_input(img_array)
            else:
                img_array /= 255.0
        except Exception:
            img_array /= 255.0

        img_array = np.expand_dims(img_array, axis=0)
        prediction = hair_type_model.predict(img_array)
        prediction_idx = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100

        st.image(img, caption="Uploaded Hair Image", width=400)
        st.markdown(f'<div class="prediction-box">💇 Predicted Hair Type: {HAIR_TYPE_CLASSES[prediction_idx]} ({confidence:.2f}% confidence)</div>', unsafe_allow_html=True)

        if st.checkbox("Show confidence details"):
            confidence_table = pd.DataFrame({
                "Hair Type": HAIR_TYPE_CLASSES,
                "Confidence (%)": (prediction[0] * 100).round(2)
            })
            st.dataframe(confidence_table)

    elif hair_img_file:
        st.warning("⚠️ Hair type model is not loaded.")

    st.markdown("## 💊 Hair Conditions and Treatments")
    hair_disease_table = pd.DataFrame({
        "Disease Name": ["Alopecia Areata", "Contact Dermatitis", "Folliculitis", "Head Lice", "Lichen Planus",
                         "Male Pattern Baldness", "Psoriasis", "Seborrheic Dermatitis", "Telogen Effluvium", "Tinea Capitis"],
        "Treatment 1": ["Corticosteroids", "Topical corticosteroids", "Warm compresses", "OTC lice meds", "Topical corticosteroids",
                        "Minoxidil", "Topical corticosteroids", "Topical antifungal", "Observation", "Oral antifungal meds"],
        "Treatment 2": ["Minoxidil", "Antihistamines", "Topical antibiotics", "Fine-toothed comb", "Antimalarial drugs",
                        "Finasteride", "Light therapy", "Topical corticosteroids", "Iron supplements", "Topical antifungal shampoo"],
        "Treatment 3": ["Immunosuppressants", "Avoid irritants", "Oral antibiotics", "Wash bedding & clothing", "Systemic corticosteroids",
                        "Hair transplant", "Systemic medications", "Selenium sulfide shampoo", "Stress management", "Selenium sulfide shampoo"]
    })
    st.dataframe(hair_disease_table)
