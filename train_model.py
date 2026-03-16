import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load the dataset
data = pd.read_csv("dataset.csv")

# Select features (symptoms)
X = data[['fever','cough','fatigue','headache','sore_throat','nausea','body_pain','runny_nose']]

# Target column
y = data['disease']

# Create model
model = DecisionTreeClassifier()

# Train the model
model.fit(X, y)

# Save the trained model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully with upgraded dataset!")