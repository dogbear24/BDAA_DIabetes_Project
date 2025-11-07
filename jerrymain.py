import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==========================
# 1. LOAD DATA
# ==========================
df = pd.read_csv("data.csv")

# Target
y = df["diagnosed_diabetes"]

# Features (exclude target and other derived columns)
X = df.drop(columns=["diagnosed_diabetes","diabetes_stage","diabetes_risk_score"])

# ==========================
# 2. CATEGORICAL AND NUMERIC
# ==========================
categorical_cols = ["gender", "ethnicity", "education_level",
                    "income_level", "employment_status", "smoking_status"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# ==========================
# 3. PIPELINE
# ==========================
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[("categorical", categorical_transformer, categorical_cols)],
    remainder="passthrough"
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
    verbose=1
)

# Full pipeline
model = Pipeline(steps=[("preprocessor", preprocessor),
                        ("classifier", rf_model)])

# ==========================
# 4. TRAIN
# ==========================
model.fit(X_train, y_train)

# ==========================
# 5. EVALUATE
# ==========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print(classification_report(y_test, y_pred))

# ==========================
# 6. SAVE MODEL
# ==========================
joblib.dump(model, "rf_diabetes_model.pkl")
print("Model saved as rf_diabetes_model.pkl")