"""
--------------------------------------------------------------
High-Value Transaction Detection - Data Preprocessing
--------------------------------------------------------------
 Author: Sachin Chhetri
 Date: 3rd, February 2025
 Description:
    - This script preprocesses raw transaction data for model training.
    - It extracts date-based features, calculates rolling statistics,
      encodes categorical variables, and prepares data for machine learning.

 Output:
    - Generates `final_preprocessed_dataset.csv` for training.
"""


import pandas as pd
import numpy as np

# Load Dataset
file_path = "Mockdata/mockdata.csv"
df = pd.read_csv(file_path)

print("\n🔍 Initial Dataset Info:")
print(df.info())
print("\n📊 Initial Data Preview:\n", df.head())

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Extract date components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Feature Engineering
## Absolute transaction amount
df['abs_transaction_amount'] = df['transaction_amount'].abs()

## Define high-value transactions (Threshold: > $5000)
df['high_value_transaction'] = (df['transaction_amount'].abs() > 5000).astype(int)

## Rolling Transaction Mean (last 5 transactions per account)
df['rolling_mean_transaction'] = df.groupby("account_holder")["abs_transaction_amount"].transform(lambda x: x.rolling(5, min_periods=1).mean())

## Transaction Count per Account Holder
df["transaction_count"] = df.groupby("account_holder")["transaction_id"].transform("count")

## Standard Deviation of last 5 transactions
df['rolling_std_transaction'] = df.groupby("account_holder")["abs_transaction_amount"].transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))

## Time since last transaction (in days)
df['time_since_last_transaction'] = df.groupby("account_holder")["date"].diff().dt.days.fillna(0)

## Average transaction amount per account
df['avg_transaction_per_account'] = df.groupby("account_holder")["abs_transaction_amount"].transform("mean")

## Percentage change from the previous transaction
df['percentage_change'] = df.groupby("account_holder")["abs_transaction_amount"].pct_change().fillna(0)

## Ratio of transaction amount to account average
df['transaction_ratio_to_avg'] = df['abs_transaction_amount'] / df['avg_transaction_per_account']

# Encode categorical variables
df['transaction_type_encoded'] = df['transaction_type'].astype('category').cat.codes
df = pd.get_dummies(df, columns=['transaction_mode'])

# Drop Unnecessary Columns
drop_columns = ['transaction_id', 'date', 'account_holder', 'account_number', 'merchant_name', 'transaction_amount']
df.drop(columns=drop_columns, inplace=True, errors='ignore')

print("\n✅ Final Data Info:")
print(df.info())
print("\n📊 Final Data Preview:\n", df.head())

# Save Preprocessed Dataset
df.to_csv("Mockdata/final_preprocessed_dataset.csv", index=False)
print("\n✅ Preprocessing Complete! File saved as 'Mockdata/final_preprocessed_dataset.csv'.")