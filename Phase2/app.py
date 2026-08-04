import streamlit as st
import joblib
import pandas as pd
import numpy as np
from utils import format_value
model = joblib.load("models/player_values.joblib")

st.title("Football Player Value Predictor")
age = st.number_input("Age", 14, 50, 27)
overall = st.number_input("Overall Rating", 50, 99, 88)
potential = st.number_input("Potential", overall, 99, 90)

age_df = pd.DataFrame({
    "age": [age]})
age_scaler = joblib.load("models/age_scaler.joblib")
scaled_age = age_scaler.transform(age_df)[0][0]

if st.button("Predict Market Value"):
    features = pd.DataFrame({
        "age":[scaled_age],
        "overall":[overall],
        "potential":[potential]
        })

    prediction = model.predict(features)
    value = format_value(np.exp(prediction[0])-1)
    st.success(value)