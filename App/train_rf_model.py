"""
--------------------------------------------------------------
High-Value Transaction Detection - Model Training
--------------------------------------------------------------
 Author: Sachin Chhetri
 Date: 3rd, February 2025 
 Description:
    - This script trains a Random Forest model to classify transactions.
    - It applies feature engineering, hyperparameter tuning, and class balancing.
    - The trained model is saved as `optimized_random_forest.pkl`.

 Output:
    - Saves the trained model and scaler for future use.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, auc
from imblearn.under_sampling import RandomUnderSampler
import joblib

# Load Preprocessed Dataset
file_path = "Mockdata/final_preprocessed_dataset.csv"
df = pd.read_csv(file_path)

# Ensure 'transaction_amount_binned' exists
if 'abs_transaction_amount' in df.columns:
    df['transaction_amount_binned'] = pd.qcut(df['abs_transaction_amount'], q=5, labels=False, duplicates='drop')
else:
    raise KeyError("'abs_transaction_amount' column is missing from the dataset. Cannot create 'transaction_amount_binned'.")

# Feature Engineering - Adding Rolling Statistics
df['avg_transaction_per_category'] = df.groupby('transaction_category')['transaction_amount_binned'].transform('mean')
df['rolling_avg_transaction'] = df['transaction_amount_binned'].rolling(window=3, min_periods=1).mean()

# Split into Training and Testing Data (80-20 split)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv("Mockdata/training_data.csv", index=False)
test_df.to_csv("Mockdata/testing_data.csv", index=False)

# Load Training Data
train_df = pd.read_csv("Mockdata/training_data.csv")

# Convert Categorical Features to Numeric
categorical_cols = train_df.select_dtypes(include=['object']).columns
train_df[categorical_cols] = train_df[categorical_cols].astype('category').apply(lambda x: x.cat.codes)

# Feature Selection
selected_features = [
    'transaction_type', 'transaction_category', 'month', 'day',
    'transaction_amount_binned', 'rolling_avg_transaction',
    'avg_transaction_per_category', 'time_since_last_transaction', 'transaction_ratio_to_avg',
    'percentage_change', 'transaction_type_encoded', 'transaction_mode_Bank Transfer',
    'transaction_mode_Cash', 'transaction_mode_Credit Card', 'transaction_mode_Crypto', 'transaction_mode_PayPal'
]

# Ensure selected features exist
existing_features = [col for col in selected_features if col in train_df.columns]
train_df = train_df[existing_features + ['high_value_transaction']]

# Define Features and Target
X_train = train_df.drop(columns=['high_value_transaction'])
y_train = train_df['high_value_transaction']

# Apply Min-Max Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Apply Undersampling for Class Balancing
undersample = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = undersample.fit_resample(X_train_scaled, y_train)

# Hyperparameter Optimization
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [7, 10],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)
print(f"\n✅ Best Parameters: {grid_search.best_params_}")

# Train Optimized Random Forest Model
rf_model = RandomForestClassifier(**grid_search.best_params_, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Save Model and Scaler
joblib.dump(rf_model, "Models/optimized_random_forest.pkl")
joblib.dump(scaler, "Models/scaler.pkl")
print("\nModel and scaler saved successfully.")

# Load Testing Data (Completely Unseen by Model)
test_df = pd.read_csv("Mockdata/testing_data.csv")

# Convert Categorical Features to Numeric (Using Training Encodings)
test_df[categorical_cols] = test_df[categorical_cols].astype('category').apply(lambda x: x.cat.codes)

# Ensure test dataset has the same features as train dataset
for col in existing_features:
    if col not in test_df.columns:
        test_df[col] = 0  # Add missing features as 0

test_df = test_df[existing_features + ['high_value_transaction']]
X_test = test_df.drop(columns=['high_value_transaction'])
y_test = test_df['high_value_transaction']

# Apply Scaling to Test Data
scaler = joblib.load("Models/scaler.pkl")
X_test_scaled = scaler.transform(X_test)

# Load Model and Predict
rf_model = joblib.load("Models/optimized_random_forest.pkl")
y_pred = rf_model.predict(X_test_scaled)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Random Forest Accuracy on Unseen Data: {accuracy:.4f}\n")
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'High-Value'], yticklabels=['Normal', 'High-Value'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Random Forest Confusion Matrix on Unseen Data')
plt.show()
