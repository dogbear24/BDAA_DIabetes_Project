import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# model selection and validation
from sklearn.model_selection import train_test_split
# estimators
from sklearn.ensemble import RandomForestRegressor
# see what this is
from sklearn.metrics import mean_squared_error, r2_squared

filepath = "diabetes_dataset.csv"
df = pd.read_csv(filepath)

df = df.drop_duplicates()
# drop rows with missing values
dr = df.dropna()
# OR fill missing values
df = df.fillna("Unknown")

print(df.head())
print(df.shape)

plt.scatter(df['age'], df['alcohol_consumption_per_week'])
plt.show()