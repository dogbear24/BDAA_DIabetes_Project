import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# 1. LOAD DATA AND MODEL
# ==========================
df = pd.read_csv("data.csv")
model = joblib.load("rf_diabetes_model.pkl")

# ==========================
# 2. DASHBOARD TITLE
# ==========================
st.title("Diabetes Prediction Dashboard")
st.markdown("Interactive dashboard for exploring diabetes data and making predictions.")

# ==========================
# 3. DATA EXPLORATION
# ==========================
st.header("Data Overview")
if st.checkbox("Show raw data"):
    st.write(df.head())

# --- Class Distribution ---
st.subheader("Class Distribution")
fig, ax = plt.subplots()
sns.countplot(x="diagnosed_diabetes", data=df, palette="Set2", ax=ax)
ax.set_xlabel("Diagnosed Diabetes")
ax.set_ylabel("Count")
st.pyplot(fig)

# --- Correlation Heatmap ---
st.subheader("Correlation Heatmap")
numeric_cols = df.select_dtypes(include=np.number).columns
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
st.pyplot(fig)

# --- Lab Feature Pairplot ---
st.subheader("Lab Feature Pairplot")
lab_features = ["hba1c", "glucose_fasting", "glucose_postprandial", "insulin_level"]
st.write("Pairplot by Diabetes Status")
pairplot_fig = sns.pairplot(df, vars=lab_features, hue="diagnosed_diabetes", corner=True)
st.pyplot(pairplot_fig.fig)

# ==========================
# 4. FEATURE IMPORTANCE
# ==========================
st.header("Feature Importances")
categorical_cols = ["gender", "ethnicity", "education_level",
                    "income_level", "employment_status", "smoking_status"]
numeric_cols = [col for col in df.columns if col not in categorical_cols + ["diagnosed_diabetes","diabetes_stage","diabetes_risk_score"]]

feature_names = list(model.named_steps["preprocessor"]
                     .transformers_[0][1]
                     .get_feature_names_out(categorical_cols)) + numeric_cols

importances = model.named_steps["classifier"].feature_importances_
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x=feature_importances.head(10).values,
            y=feature_importances.head(10).index,
            palette="viridis",
            dodge=False,
            ax=ax)
ax.set_title("Top 10 Feature Importances")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
st.pyplot(fig)

# ==========================
# 5. USER INPUT FOR PREDICTION
# ==========================
st.header("Predict Diabetes for a New Patient")
st.markdown("Fill in patient information:")

def user_input_features():
    gender = st.selectbox("Gender", df["gender"].unique())
    ethnicity = st.selectbox("Ethnicity", df["ethnicity"].unique())
    education_level = st.selectbox("Education Level", df["education_level"].unique())
    income_level = st.selectbox("Income Level", df["income_level"].unique())
    employment_status = st.selectbox("Employment Status", df["employment_status"].unique())
    smoking_status = st.selectbox("Smoking Status", df["smoking_status"].unique())
    
    # Numeric inputs
    hba1c = st.number_input("HbA1c", min_value=3.0, max_value=20.0, value=5.5, step=0.1)
    glucose_fasting = st.number_input("Fasting Glucose", min_value=50, max_value=400, value=100)
    glucose_postprandial = st.number_input("Postprandial Glucose", min_value=50, max_value=500, value=140)
    insulin_level = st.number_input("Insulin Level", min_value=0, max_value=300, value=80)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    
    data = {
        "gender": gender,
        "ethnicity": ethnicity,
        "education_level": education_level,
        "income_level": income_level,
        "employment_status": employment_status,
        "smoking_status": smoking_status,
        "hba1c": hba1c,
        "glucose_fasting": glucose_fasting,
        "glucose_postprandial": glucose_postprandial,
        "insulin_level": insulin_level,
        "age": age,
        "bmi": bmi
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.subheader("Prediction Result")
    st.write("Diabetes Diagnosed" if prediction == 1 else "No Diabetes")