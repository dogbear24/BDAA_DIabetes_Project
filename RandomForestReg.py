import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Open data table to df
filepath = "diabetes_dataset.csv"
df = pd.read_csv(filepath)

# Clean data
df = df.drop_duplicates()
df = df.dropna()
df = df.fillna("Unknown")

# Prepare features for encoding
categorical_features = ['gender', 'ethnicity', 'education_level', 'income_level', 
                       'employment_status', 'smoking_status']

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
encoders = {}
for feature in categorical_features:
    encoders[feature] = LabelEncoder()
    df[feature] = encoders[feature].fit_transform(df[feature])

# Select features for prediction
feature_columns = ['age', 'gender', 'ethnicity', 'education_level', 'income_level',
                  'employment_status', 'smoking_status', 'alcohol_consumption_per_week',
                  'physical_activity_minutes_per_week', 'diet_score', 'sleep_hours_per_day',
                  'screen_time_hours_per_day', 'family_history_diabetes', 'hypertension_history',
                  'cardiovascular_history', 'bmi', 'waist_to_hip_ratio', 'systolic_bp',
                  'diastolic_bp', 'heart_rate', 'cholesterol_total', 'hdl_cholesterol',
                  'ldl_cholesterol', 'triglycerides', 'glucose_fasting', 'glucose_postprandial',
                  'insulin_level']

# Define target variable (we'll predict diabetes_risk_score)
target = 'diabetes_risk_score'

# Prepare features (X) and target (y)
X = df[feature_columns]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Fit the model
rf_model.fit(X_train, y_train)

# Make predictions
predictions = rf_model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = sk.metrics.r2_score(y_test, predictions)

print("\nModel Performance Metrics:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
