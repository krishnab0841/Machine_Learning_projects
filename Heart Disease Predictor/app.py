# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import joblib 

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.title("Heart diesease predictor by Krishna")
st.markdown("Provide the Following Details")

Age = st.slider("Age",18,100,40)
Sex = st.selectbox("Sex",["Male","Female"])
Chest_Pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
Resting_BP = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
Cholestoral = st.number_input("Cholestoral (mm Hg)", 80, 200, 120)
Fasting_BS = st.selectbox("Fasting Blood Sugar > 120mg/dl",["Yes","No"])
Resting_ECG = st.selectbox("Resting ECG",["Normal","ST-T wave abnormality","Left Ventricular Hypertrophy"])
Max_HR = st.slider("Max Heart Rate", 60, 220, 150)
Excercise_Angina = st.selectbox("Excercise Induced Angina",["Yes","No"])
Oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["UP", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        "Age" : Age,
        "Sex_"+ Sex : 1,
        "Chest PainType" + Chest_Pain:1,
        "Resting BP" : Resting_BP,
        "Cholestoral" : Cholestoral,
        "Fasting Blood Sugar" : Fasting_BS,
        "Resting ECG_" + Resting_ECG: 1,
        "Max Heart Rate" : Max_HR,
        "Excercise Induced Angina" + Excercise_Angina: 1,
        "Oldpeak" : Oldpeak,
        "ST_Slope_" + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    
    input_df = input_df[expected_columns]
    
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    
    if prediction == 1:
        st.error("Heart diesease detected")
    else:
        st.success("No heart diesease detected")