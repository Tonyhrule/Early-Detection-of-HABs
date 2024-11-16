import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = "Dataset.xlsx"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURES = ['Temperature', 'Salinity', 'UVB']
TARGET = 'ChlorophyllaFlor'

def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    print("Loading dataset...")
    return pd.read_excel(data_path)

def clean_data(data):
    print("Cleaning data: removing rows with NaN values...")
    original_shape = data.shape
    data = data.dropna()
    print(f"Original shape: {original_shape}, Cleaned shape: {data.shape}")
    return data

def preprocess_data(data):
    print("Preprocessing data...")
    X = data[FEATURES]
    y = data[TARGET]
    print("Imputing missing values using mean...")
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    if len(y.shape) > 1:
        print("Flattening target variable...")
        y = y.ravel()
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Preprocessing complete!")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def evaluate_basic_model(X_train, X_test, y_train, y_test):
    print("\nTraining and evaluating a basic Ridge regression model...")
    model = Ridge(alpha=10.0) 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nEvaluation Metrics:")
    print(f"  MSE: {mse}")
    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")
    print(f"  R² Score: {r2}")
    return mse, rmse, mae, r2

def save_data(X_train, X_test, y_train, y_test, scaler, output_dir):
    print("Saving processed data...")
    joblib.dump((X_train, X_test, y_train, y_test), os.path.join(output_dir, 'processed_data.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print("Processed data saved successfully!")

def main():
    data = load_data(DATA_PATH)
    if not all(col in data.columns for col in FEATURES + [TARGET]):
        raise ValueError(f"Dataset does not contain required columns: {FEATURES + [TARGET]}")
    data = clean_data(data)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    evaluate_basic_model(X_train, X_test, y_train, y_test)
    save_data(X_train, X_test, y_train, y_test, scaler, OUTPUT_DIR)

if __name__ == "__main__":
    main()
