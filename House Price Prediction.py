# ================================
# House Price Prediction Project
# ================================

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ================================
# STEP 1: CHECK CURRENT DIRECTORY
# ================================
print("Current Working Directory:", os.getcwd())
print("Files in Directory:", os.listdir())

# ================================
# STEP 2: LOAD DATASET (USE FULL PATH)
# ================================
df = pd.read_csv("Task 3/housing_dataset.csv")

print("\nDataset Loaded Successfully!\n")
print(df.head())

# ================================
# STEP 3: DATA PREPROCESSING
# ================================
print("\nMissing Values:\n", df.isnull().sum())

# Fill missing values if any
df.fillna(df.mean(), inplace=True)

# ================================
# STEP 4: FEATURES & TARGET
# ================================
X = df.drop("MEDV", axis=1)   # Features
y = df["MEDV"]                # Target

# ================================
# STEP 5: TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# STEP 6: SCALING
# ================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# STEP 7: TRAIN MODEL
# ================================
model = LinearRegression()
model.fit(X_train, y_train)

# ================================
# STEP 8: PREDICTION
# ================================
y_pred = model.predict(X_test)

# ================================
# STEP 9: EVALUATION
# ================================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MSE:", mse)
print("R2 Score:", r2)

# ================================
# STEP 10: SAMPLE PREDICTION
# ================================
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)

print("\nSample Prediction:", prediction[0])