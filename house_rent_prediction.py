# house_rent_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

# --- Data Generation (for demonstration purposes) ---
# Since we don't have the actual CSV file, we'll create a synthetic dataset
# that mimics the features described in the article.

def generate_synthetic_data():
    """Generates a synthetic dataset for house rent prediction."""
    data = {
        'BHK': np.random.randint(1, 6, 100),
        'Size': np.random.randint(500, 2500, 100),
        'Area Type': np.random.choice(['Super Area', 'Carpet Area', 'Built Area'], 100),
        'City': np.random.choice(['Mumbai', 'Chennai', 'Bangalore', 'Delhi', 'Hyderabad'], 100),
        'Furnishing Status': np.random.choice(['Semi-Furnished', 'Unfurnished', 'Furnished'], 100),
        'Rent': np.random.randint(5000, 50000, 100) + np.random.randint(1, 6, 100) * 10000
    }
    df = pd.DataFrame(data)
    
    # Adjust rent based on other features to make the data more realistic
    df['Rent'] = df['Rent'] + df['BHK'] * 5000 + df['Size'] * 5
    return df

# Create the synthetic DataFrame
df = generate_synthetic_data()

# --- Exploratory Data Analysis (EDA) ---
print("--- Initial Data Snapshot ---")
print(df.head())
print("\n--- Data Information ---")
print(df.info())
print("\n--- Descriptive Statistics ---")
print(df.describe())

# Visualize the distribution of the target variable 'Rent'
plt.figure(figsize=(10, 6))
sns.histplot(df['Rent'], kde=True)
plt.title('Distribution of House Rent')
plt.xlabel('Rent')
plt.ylabel('Frequency')
plt.show()

# Visualize rent based on 'BHK'
plt.figure(figsize=(10, 6))
sns.boxplot(x='BHK', y='Rent', data=df)
plt.title('Rent vs. BHK')
plt.xlabel('BHK')
plt.ylabel('Rent')
plt.show()

# --- Data Preprocessing and Feature Engineering ---

# Identify categorical and numerical features
categorical_features = ['Area Type', 'City', 'Furnishing Status']
numerical_features = ['BHK', 'Size']

# Create a column transformer to apply one-hot encoding to categorical features
# This automates the process of transforming categorical data into a format
# that machine learning models can understand.
preprocessor = make_column_transformer(
    (OneHotEncoder(), categorical_features),
    remainder='passthrough'
)

# Apply the transformations
X_processed = preprocessor.fit_transform(df.drop('Rent', axis=1))
y = df['Rent']

# --- Model Training ---

# Split the data into training and testing sets
# We use a 80/20 split, with 80% for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Choose and train a model. The article mentioned several; we'll use Linear Regression.
model = LinearRegression()
model.fit(X_train, y_train)

# --- Model Evaluation and Prediction ---

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the R-squared score to evaluate the model's performance
# R-squared measures how well the model's predictions fit the actual data.
# A score of 1.0 means a perfect fit.
r2 = r2_score(y_test, y_pred)
print("\n--- Model Evaluation ---")
print(f"R-squared Score: {r2:.2f}")

# Example of a new prediction
# To make a prediction on new data, it also needs to be transformed
# The data must have the same columns and order as the training data
new_data = pd.DataFrame([
    [3, 1500, 'Super Area', 'Bangalore', 'Furnished']
], columns=['BHK', 'Size', 'Area Type', 'City', 'Furnishing Status'])

# Apply the same transformations to the new data
new_data_processed = preprocessor.transform(new_data)

# Make the prediction
predicted_rent = model.predict(new_data_processed)
print("\n--- Prediction for a New House ---")
print(f"Predicted Rent for a new house: ${predicted_rent[0]:.2f}")

print("\nScript finished.")
