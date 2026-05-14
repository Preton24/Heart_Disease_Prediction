import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

st.title("Heart Disease Prediction System", anchor=False)

st.write("HCAI-Based Healthcare Prediction for SDG 3")

# -----------------------------------
# User Inputs Setup
# -----------------------------------

# Initialize session state for inputs
defaults = {
    'age': 40,
    'sex': "M",
    'chest_pain': "ATA",
    'resting_bp': 120,
    'cholesterol': 200,
    'fasting_bs': 0,
    'max_hr': 150,
    'exercise_angina': "N",
    'oldpeak': 1.0
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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

def load_demo_2():
    st.session_state.age = 40
    st.session_state.sex = "F"
    st.session_state.chest_pain = "ATA"
    st.session_state.resting_bp = 120
    st.session_state.cholesterol = 200
    st.session_state.fasting_bs = 0
    st.session_state.max_hr = 170
    st.session_state.exercise_angina = "N"
    st.session_state.oldpeak = 0.0

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Demo Data", anchor=False)
    st.write("Load realistic patient profiles:")
    st.button("Demo 1: High Risk Profile", on_click=load_demo_1, use_container_width=True)
    st.button("Demo 2: Low Risk Profile", on_click=load_demo_2, use_container_width=True)

with col1:
    age = st.number_input("Age", 1, 100, key="age")
    sex = st.selectbox("Sex", ["M", "F"], key="sex")
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"], key="chest_pain")
    resting_bp = st.number_input("Resting Blood Pressure", 50, 250, key="resting_bp")
    cholesterol = st.number_input("Cholesterol", 50, 700, key="cholesterol")
    fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1], key="fasting_bs")
    max_hr = st.number_input("Maximum Heart Rate", 60, 250, key="max_hr")
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"], key="exercise_angina")
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, key="oldpeak")

# -----------------------------------
# Encode Inputs
# -----------------------------------

sex_value = 1 if sex == "M" else 0

cp_map = {
    "ATA": 0,
    "NAP": 1,
    "ASY": 2,
    "TA": 3
}

chest_pain_value = cp_map[chest_pain]

exercise_value = 1 if exercise_angina == "Y" else 0

# Dummy/default values for remaining features
resting_ecg = 1
st_slope = 1

# -----------------------------------
# Prediction
# -----------------------------------

if st.button("Predict"):

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

    st.subheader("Prediction Results", anchor=False)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

    st.write(f"**Risk Probability:** {risk_prob * 100:.1f}%")

    st.subheader("Risk Assessment", anchor=False)
    if risk_prob < 0.3:
        st.info("🟢 **Low Risk**: No immediate concern. Keep maintaining a healthy lifestyle!")
    elif risk_prob < 0.7:
        st.warning("🟡 **Moderate Risk**: You have some risk factors. Consider consulting a doctor for a routine checkup.")
    else:
        st.error("🔴 **High Risk**: High likelihood of heart disease. Please consult a doctor immediately for a thorough evaluation.")