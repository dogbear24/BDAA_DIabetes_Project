import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

filepath = "data.csv"

df = pd.read_csv(filepath)

df = df.dropna()

x = df.drop(["diagnosed_diabetes","diabetes_stage"], axis=1)
y = df["diagnosed_diabetes"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(xtrain, ytrain)

ypred = rf_classifier.predict(xtest)

accuracy = accuracy_score(ytest, ypred)
classification_rep = classification_report(ytest, ypred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)
###



