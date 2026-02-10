
import streamlit as st
import joblib
import numpy as np

model = joblib.load("model/model.pkl")

st.title("Student Mental Health Predictor")

age = st.slider("Age",18,30)
study = st.slider("Study Hours",0,12)
sleep = st.slider("Sleep Hours",0,12)
marks = st.slider("Academic Performance",0,100)
alcohol = st.selectbox("Alcoholic",["No","Yes"])

alcohol = 1 if alcohol=="Yes" else 0

if st.button("Predict"):
    data = np.array([[age,study,sleep,marks,alcohol]])
    pred = model.predict(data)[0]
    result = ["High Risk","Low Risk","Medium Risk"]
    st.success(result[pred])
