import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Paths
DATA_PATH = "Dataset.xlsx"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURES = ['Temperature', 'Salinity', 'UVB']
TARGET = 'ChlorophyllaFlor'

def load_data(data_path):
    """Loads the dataset from an Excel file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    print("Loading dataset...")
    return pd.read_excel(data_path)

def preprocess_data(data):
    """Preprocesses the dataset."""
    print("Preprocessing data...")

    # Separate features and target
    X = data[FEATURES]
    y = data[TARGET]

    # Handle missing values with median imputation
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Ensure target is 1D
    if len(y.shape) > 1:
        print("Flattening target variable...")
        y = y.ravel()

    # Train-test split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Preprocessing complete!")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_data(X_train, X_test, y_train, y_test, scaler, output_dir):
    """Saves processed data and scaler."""
    print("Saving processed data...")
    joblib.dump((X_train, X_test, y_train, y_test), os.path.join(output_dir, 'processed_data.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print("Processed data saved successfully!")

def main():
    """Main function for preprocessing."""
    # Load data
    data = load_data(DATA_PATH)

    # Validate data structure
    if not all(col in data.columns for col in FEATURES + [TARGET]):
        raise ValueError(f"Dataset does not contain required columns: {FEATURES + [TARGET]}")

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    # Save preprocessed data
    save_data(X_train, X_test, y_train, y_test, scaler, OUTPUT_DIR)

if __name__ == "__main__":
    main()
