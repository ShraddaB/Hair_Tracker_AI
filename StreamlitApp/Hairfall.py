import streamlit as st
import joblib
import os

# Path to Hairfall model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xgb_model_path = os.path.join(BASE_DIR, 'AI_Models', 'Hairfall_Prediction', 'XGB.joblib')

# Load Hairfall model
try:
    hairfall_model = joblib.load(xgb_model_path)
    print("Hairfall model loaded successfully.")
except Exception as e:
    hairfall_model = None
    print(f"Error loading hairfall model: {e}")

def run_page():
    st.markdown("## Hairfall Prediction")
    if hairfall_model:
        pressure_level = st.selectbox("Pressure Level", [0,1,2,3])
        stress_level = st.selectbox("Stress Level", [0,1,2,3])
        hair_grease = st.selectbox("Hair Grease Level", [3.0,1.0,2.0,4.0,5.0])
        dandruff = st.selectbox("Dandruff Level", [1.0,2.0])

        if st.button("Predict"):
            inputs = [[pressure_level, stress_level, hair_grease, dandruff]]
            prediction = hairfall_model.predict(inputs)[0]
            message = "might experience hairfall." if prediction==1 else "might not experience hairfall."
            st.success(f"Prediction: The person {message}")
    else:
        st.warning("Hairfall model is missing or could not be loaded.")
