<<<<<<< HEAD:jasonmain.py
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



=======
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

filepath = "data.csv"
df = pd.read_csv(filepath)

# Define features (X) and target (y)
X = df.drop(columns=["diagnosed_diabetes","diabetes_stage"])  # features
y = df["diagnosed_diabetes"]                 # target

# Identify categorical and numeric columns
categorical_cols = ["gender", "ethnicity", "education_level", "income_level", 
                    "employment_status", "smoking_status", "diabetes_stage"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_transformer, categorical_cols)
    ],
    remainder="passthrough"  # keep numeric columns as-is
)

# Split into training/testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced"  # optional if classes are imbalanced
)

# Combine preprocessing + model in a pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", rf_model)
])

model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Extract feature names from the preprocessing step
feature_names = list(model.named_steps["preprocessor"]
                     .transformers_[0][1]
                     .get_feature_names_out(categorical_cols)) + numeric_cols

importances = model.named_steps["classifier"].feature_importances_
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print(feature_importances.head(10))
>>>>>>> b749417d8b58642cc6bf8a5c2d017ede3b795966:main.py
