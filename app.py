import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    /* App background */
    .stApp {
        background-color: #fdf7f7;
        font-family: 'Inter', sans-serif;
    }
    /* Custom button styling */
    div.stButton > button:first-child {
        background-color: #800020;
        color: white;
        border-radius: 8px;
        transition: all 0.3s;
        border: none;
        font-weight: 600;
    }
    div.stButton > button:first-child:hover {
        background-color: #5c0017;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    /* Headers */
    h1 {
        color: #4a0e1b;
        font-family: 'Playfair Display', serif !important;
    }
    h2, h3, h4, h5, h6 {
        color: #4a0e1b;
        font-family: 'Inter', sans-serif;
    }
    /* Cards */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(128, 0, 32, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------
# Load Dataset
# -----------------------------------

df = pd.read_csv("heart.csv")

# Encode categorical columns
encoder = LabelEncoder()

for column in df.columns:
    if not pd.api.types.is_numeric_dtype(df[column]):
        df[column] = encoder.fit_transform(df[column])

# -----------------------------------
# Features and Target
# -----------------------------------

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# -----------------------------------
# Train Model
# -----------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# -----------------------------------
# Streamlit UI
# -----------------------------------

header_col1, header_col2 = st.columns([10, 1])

with header_col1:
    st.title("Heart Disease Prediction System", anchor=False)
    st.markdown("<p style='font-size: 1.2rem; color: #7f8c8d; font-weight: 500;'>HCAI-Based Healthcare Prediction for SDG 3</p>", unsafe_allow_html=True)

with header_col2:
    st.markdown(
        """
        <div style="text-align: right; padding-top: 15px;">
            <a href="https://docs.google.com/spreadsheets/d/1OMH70yg-iTh3Qe73MCsKzoc9ndNCEMadEgkBBWJ5C60/edit?usp=sharing" target="_blank" title="Preview Dataset" style="text-decoration: none; display: inline-block; text-align: center;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/3/30/Google_Sheets_logo_%282014-2020%29.svg" width="40" style="transition: transform 0.2s;" onmouseover="this.style.transform='scale(1.1)'" onmouseout="this.style.transform='scale(1)'">
                <div style="font-size: 0.75rem; color: #7f8c8d; margin-top: 4px; font-weight: 500;">Dataset Preview</div>
            </a>
        </div>
        """, 
        unsafe_allow_html=True
    )

st.markdown("---")

# -----------------------------------
# User Inputs Setup
# -----------------------------------

# Session state initialization is handled automatically by Streamlit

def load_demo_1():
    st.session_state.age = 65
    st.session_state.sex = "M"
    st.session_state.chest_pain = "ASY"
    st.session_state.resting_bp = 160
    st.session_state.cholesterol = 280
    st.session_state.fasting_bs = 1
    st.session_state.max_hr = 110
    st.session_state.exercise_angina = "Y"
    st.session_state.oldpeak = 2.5

def load_demo_3():
    st.session_state.age = 40
    st.session_state.sex = "F"
    st.session_state.chest_pain = "ATA"
    st.session_state.resting_bp = 120
    st.session_state.cholesterol = 200
    st.session_state.fasting_bs = 0
    st.session_state.max_hr = 170
    st.session_state.exercise_angina = "N"
    st.session_state.oldpeak = 0.0

def load_demo_2():
    st.session_state.age = 55
    st.session_state.sex = "M"
    st.session_state.chest_pain = "NAP"
    st.session_state.resting_bp = 135
    st.session_state.cholesterol = 240
    st.session_state.fasting_bs = 0
    st.session_state.max_hr = 140
    st.session_state.exercise_angina = "N"
    st.session_state.oldpeak = 1.0

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Demo Data", anchor=False)
    st.write("Load realistic patient profiles:")
    st.button("Demo 1: High Risk Profile", on_click=load_demo_1, use_container_width=True)
    st.button("Demo 2: Moderate Risk Profile", on_click=load_demo_2, use_container_width=True)
    st.button("Demo 3: Low Risk Profile", on_click=load_demo_3, use_container_width=True)

with col1:
    st.subheader("Patient Details", anchor=False)
    # Use nested columns to make it look better
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=100, value=None, key="age")
        resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=None, key="resting_bp")
        max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=250, value=None, key="max_hr")
    with c2:
        sex = st.selectbox("Sex", ["M", "F"], index=None, key="sex")
        cholesterol = st.number_input("Cholesterol", min_value=50, max_value=700, value=None, key="cholesterol")
        exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"], index=None, key="exercise_angina")
    with c3:
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"], index=None, key="chest_pain")
        fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1], index=None, key="fasting_bs")
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=None, key="oldpeak")

# -----------------------------------
# Encode Inputs & Prediction
# -----------------------------------

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col = st.columns([5, 1])
    with btn_col:
        predict_clicked = st.button("Predict", use_container_width=True)

    if predict_clicked:
        if None in [age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, max_hr, exercise_angina, oldpeak]:
            st.warning("⚠️ Please fill out all patient details before predicting.")
            st.stop()

    sex_value = 1 if sex == "M" else 0

    cp_map = {
        "ASY": 0,
        "ATA": 1,
        "NAP": 2,
        "TA": 3
    }

    chest_pain_value = cp_map[chest_pain]

    exercise_value = 1 if exercise_angina == "Y" else 0

    # Dummy/default values for remaining features
    resting_ecg = 1
    st_slope = 1

    input_data = pd.DataFrame([[
        age,
        sex_value,
        chest_pain_value,
        resting_bp,
        cholesterol,
        fasting_bs,
        resting_ecg,
        max_hr,
        exercise_value,
        oldpeak,
        st_slope
    ]], columns=X.columns)

    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    risk_prob = probabilities[0][1]

    st.markdown("---")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.subheader("Prediction Results", anchor=False)
        if prediction[0] == 1:
            st.error("**⚠️ High Risk of Heart Disease**")
        else:
            st.success("**✅ Low Risk of Heart Disease**")
        
        st.metric(label="Risk Probability", value=f"{risk_prob * 100:.1f}%")

    with res_col2:
        st.subheader("Risk Assessment Details", anchor=False)
        if risk_prob < 0.3:
            st.info("🟢 **Low Risk**: No immediate concern. Keep maintaining a healthy lifestyle!")
        elif risk_prob < 0.7:
            st.warning("🟡 **Moderate Risk**: You have some risk factors. Consider consulting a doctor for a routine checkup.")
        else:
            st.error("🔴 **High Risk**: High likelihood of heart disease. Please consult a doctor immediately for a thorough evaluation.")