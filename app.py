import pickle

import pandas as pd
import streamlit as st

SYMPTOMS = {
    "fever": "Fever",
    "cough": "Cough",
    "fatigue": "Fatigue",
    "headache": "Headache",
    "sore_throat": "Sore throat",
    "nausea": "Nausea",
    "body_pain": "Body pain",
    "runny_nose": "Runny nose",
}

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🧠 AI Disease Prediction System")
st.write("Tick the symptoms you are experiencing:")

cols = st.columns(2)
selected = {}
for i, (key, label) in enumerate(SYMPTOMS.items()):
    selected[key] = int(cols[i % 2].checkbox(label))

if st.button("Predict Disease"):
    if sum(selected.values()) == 0:
        st.warning("Please select at least one symptom.")
    else:
        # same column names/order the model was trained on
        input_df = pd.DataFrame([selected])[list(SYMPTOMS)]
        probs = model.predict_proba(input_df)[0]
        ranking = sorted(zip(model.classes_, probs), key=lambda p: p[1], reverse=True)

        top, conf = ranking[0]
        st.success(f"Most likely: **{top}** ({conf * 100:.0f}% confidence)")

        st.write("All possibilities:")
        st.bar_chart(pd.DataFrame({"probability": dict(ranking)}))

st.write("---")
st.caption("⚠️ Educational demo trained on a small hand-made dataset. "
           "Not a medical diagnosis tool.")
