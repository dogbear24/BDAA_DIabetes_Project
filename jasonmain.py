import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

filepath = "data.csv"

df = pd.read_csv(filepath)

df = df.dropna()

x1 = df.drop(["diagnosed_diabetes","diabetes_stage"], axis=1)
y = df["diagnosed_diabetes"]

categorical_cols = x1.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'  # keep numerical columns
)

x = preprocessor.fit_transform(x1)




xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(xtrain, ytrain)

ypred = rf_classifier.predict(xtest)

accuracy = accuracy_score(ytest, ypred)
classification_rep = classification_report(ytest, ypred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)


numeric_cols = df.select_dtypes(include=['number']).columns
numeric = df[numeric_cols]

corr_matrix = numeric.corr()
plt.figure(figsize=(12,10))
sns.heatmap(numeric.corr()[['diagnosed_diabetes']].sort_values(by='diagnosed_diabetes', ascending=False),
            annot=True, cmap='coolwarm')
plt.title("COrrelation with Diagnosed Diabetes")
plt.show()

