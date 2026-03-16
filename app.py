import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("🧠 AI Disease Prediction System")
st.write("Select the symptoms you are experiencing:")

# Symptom inputs
fever = st.selectbox("Fever", [0,1])
cough = st.selectbox("Cough", [0,1])
fatigue = st.selectbox("Fatigue", [0,1])
headache = st.selectbox("Headache", [0,1])
sore_throat = st.selectbox("Sore Throat", [0,1])
nausea = st.selectbox("Nausea", [0,1])
body_pain = st.selectbox("Body Pain", [0,1])
runny_nose = st.selectbox("Runny Nose", [0,1])

if st.button("Predict Disease"):

    input_data = np.array([[fever,cough,fatigue,headache,sore_throat,nausea,body_pain,runny_nose]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    confidence = max(probability[0]) * 100

    st.success(f"Predicted Disease: {prediction[0]}")
    st.info(f"Confidence Level: {confidence:.2f}%")

st.write("---")
st.write("⚠️ This AI model is for educational purposes only and not a medical diagnosis tool.")