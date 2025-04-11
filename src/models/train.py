
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def load_processed_data():
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')

    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    return metrics

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Save model
    joblib.dump(model, 'models/random_forest_model.joblib')

    # Save metrics
    pd.DataFrame([metrics]).to_csv('models/metrics.csv', index=False)

    print(f"Model trained successfully with accuracy: {metrics['accuracy']:.4f}")
