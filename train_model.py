import pickle

import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.tree import DecisionTreeClassifier

FEATURES = ['fever', 'cough', 'fatigue', 'headache',
            'sore_throat', 'nausea', 'body_pain', 'runny_nose']

# Load the dataset
data = pd.read_csv("dataset.csv")
X = data[FEATURES]
y = data['disease']

model = DecisionTreeClassifier(random_state=42)

# Honest evaluation: with only 15 rows, a normal train/test split is too small,
# so use leave-one-out cross-validation (train on 14 rows, test on the 1 left out,
# repeat for every row).
scores = cross_val_score(model, X, y, cv=LeaveOneOut())
print(f"Leave-one-out accuracy: {scores.mean():.0%} ({int(scores.sum())}/{len(scores)} correct)")

# Train on all data and save
model.fit(X, y)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved to model.pkl")
