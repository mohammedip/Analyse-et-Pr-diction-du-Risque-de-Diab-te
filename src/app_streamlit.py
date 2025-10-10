import joblib
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler


model = joblib.load("model/best_model.joblib")


st.title("Diabetes Data Input Form")

st.write("Please enter the following details:")

# Create input fields
pregnancies = st.number_input("Pregnancies", min_value=0 , step=1)
glucose = st.number_input("Glucose", min_value=0.0, step=0.1)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, step=0.1)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, step=0.1)
insulin = st.number_input("Insulin", min_value=0.0, step=0.1)
bmi = st.number_input("BMI", min_value=0.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
age = st.number_input("Age", min_value=0, step=1)

# Button to display results
if st.button("Submit"):
    user_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    df = pd.DataFrame([user_data])

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

    prediction = model.predict(df_scaled)

    st.text(f"{prediction}")

