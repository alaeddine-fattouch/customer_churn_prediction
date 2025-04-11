
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Handle missing values
    df = df.fillna({
        'TotalCharges': df['MonthlyCharges'] * df['tenure']
    })

    # Convert categorical columns
    cat_columns = df.select_dtypes(include=['object']).columns
    for col in cat_columns:
        df[col] = df[col].astype('category')

    return df

def split_data(df, target_col='Churn'):
    # Convert target to binary
    df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})

    # Split data
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load raw data
    raw_data = load_data('data/raw/customer_data.csv')

    # Preprocess
    processed_data = preprocess_data(raw_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(processed_data)

    # Save preprocessed data
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    print("Data preprocessing completed successfully!")
