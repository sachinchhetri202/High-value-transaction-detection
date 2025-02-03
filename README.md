# High-value-transaction-detection
A machine learning model to classify high-value transactions using Random Forest.

## Introduction

Financial institutions need robust methods to flag transactions that deviate from the norm. This project leverages machine learning to identify high-value transactions that might require further investigation. It consists of three primary modules:

- **Data Preprocessing**: Cleaning and preparing raw bank transaction data.
- **Model Training**: Training a Random Forest classifier to detect high-value transactions.
- **Model Testing**: Evaluating the performance of the trained model.

## Project Structure

```plaintext
High-value-transaction-detection/
├── App/
│   ├── Models/               # Contains model artifacts or definitions
│   ├── Mockdata/             # Contains sample/mock data (this folder is ignored by Git)
│   ├── data_preprocessing.py # Script for cleaning and preprocessing the data
│   ├── train_rf_model.py     # Script to train the Random Forest model
│   ├── test_rf_model.py      # Script to evaluate the trained model
│   └── requirements.txt      # Python dependencies for the project
└── README.md                 # Project documentation
