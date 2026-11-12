# 🧠 AI Disease Prediction System

A Streamlit web app that predicts a likely illness from the symptoms a user ticks, using a Decision Tree classifier. It shows the most likely disease, its confidence, and the probability of every other option.

## Features

- Symptom checklist with 8 symptoms: fever, cough, fatigue, headache, sore throat, nausea, body pain, runny nose
- Predicts one of 5 conditions: **Flu, Cold, Migraine, Food Poisoning, COVID**
- Confidence score and a bar chart of the probability for every condition
- Model evaluation with leave-one-out cross-validation

## How it works

```
dataset.csv ─► train_model.py ─► model.pkl ─► app.py (Streamlit UI)
 symptoms (0/1)     Decision Tree     saved model      user ticks symptoms → prediction
 + disease label    + LOO accuracy
```

## Run it

```bash
pip install -r requirements.txt
python train_model.py      # optional: retrain the model and print its accuracy
streamlit run app.py       # opens the app in your browser
```

## Model performance

Leave-one-out cross-validation accuracy: **67% (10/15)**. The model is trained on 14 rows and tested on the 1 left out, repeated for every row. Leave-one-out is used because the dataset is too small for a normal train/test split.

## Limitations

- The dataset is **small and hand-made**: 15 rows, 3 per disease. It demonstrates the ML workflow; it does not reflect real clinical data.
- Real symptom-based diagnosis needs far larger datasets (for example, the public Kaggle *Disease Symptom Prediction* dataset), more symptoms, and validation by clinicians.

**Possible next steps:** train on a larger public dataset, compare models (Random Forest, Naive Bayes), and add symptom severity rather than yes/no.

> ⚠️ Educational project only. Not a medical diagnosis tool.

## Tech

Python · scikit-learn · pandas · Streamlit
