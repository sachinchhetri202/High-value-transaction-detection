"""
--------------------------------------------------------------
High-Value Transaction Detection - Model Testing
--------------------------------------------------------------
 Author: Sachin Chhetri
 Date: 3rd, February 2025
 Description:
    - This script loads the trained model and tests it with manual input.
    - It preprocesses input transactions, runs predictions, and outputs results.

 Output:
    - Displays whether the transaction is high-value and shows model confidence.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, auc
import joblib

# Load Trained Model and Scaler
model_path = "Models/optimized_random_forest.pkl"
scaler_path = "Models/scaler.pkl"
rf_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Manually input a HIGH-VALUE test transaction
manual_test_data = {
    'transaction_type': [2],  # Assume '2' is more associated with high-value transactions
    'transaction_category': [5],  # A category known for high-value transactions
    'month': [12],  # Possibly year-end large transactions
    'day': [25],  # Holiday season transactions might be higher
    'transaction_amount_binned': [4],  # High bin means large amount
    'rolling_avg_transaction': [4.5],  # High rolling average transaction amount
    'avg_transaction_per_category': [4.8],  # High category spending
    'time_since_last_transaction': [1],  # Frequent transactions could indicate business spending
    'transaction_ratio_to_avg': [1.5],  # Significantly higher than average spending
    'percentage_change': [0.20],  # Large spike in spending
    'transaction_type_encoded': [2],  # Assume '2' correlates to high-value
    'transaction_mode_Bank Transfer': [0],
    'transaction_mode_Cash': [0],
    'transaction_mode_Credit Card': [1],  # Credit card transactions could be larger
    'transaction_mode_Crypto': [1],  # Cryptocurrency could indicate larger transfers
    'transaction_mode_PayPal': [0]
}


manual_test_df = pd.DataFrame(manual_test_data)

# Scale Test Data
X_manual_test_scaled = scaler.transform(manual_test_df)

# Make Predictions
y_pred = rf_model.predict(X_manual_test_scaled)
y_pred_proba = rf_model.predict_proba(X_manual_test_scaled)[:, 1]

# Display Results
print(f"\n🔍 Predicted High-Value Transaction: {y_pred[0]}")
print(f"Prediction Probability: {y_pred_proba[0]:.4f}")
