import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Configuration Parameters
DATA_PATH = 'Dataset.xlsx'
OUTPUT_DIR = 'output'
TEST_SIZE = 0.2  # Test data size
RANDOM_STATE = 42  # Random seed for reproducibility

def load_data(data_path):
    """Loads the dataset from an Excel file."""
    print("Loading dataset...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at path: {data_path}")
    data = pd.read_excel(data_path)
    print("Dataset loaded successfully.")
    return data

def clean_data(data):
    """Cleans the dataset by removing rows with missing values in key columns."""
    print("Cleaning data by dropping rows with missing values...")
    data = data.dropna(subset=['Temperature', 'Salinity', 'UVB', 'ChlorophyllaFlor'])
    print(f"Data cleaned. Remaining rows: {len(data)}")
    return data

def preprocess_data(data):
    """Splits, imputes, and scales data for model training."""
    print("Selecting features and target...")
    X = data[['Temperature', 'Salinity', 'UVB']]
    y = data['ChlorophyllaFlor']

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print("Data split complete. Training set size:", X_train.shape[0])

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling complete.")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_data(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, output_dir):
    """Saves processed data and scaler to disk."""
    os.makedirs(output_dir, exist_ok=True)
    print("Saving processed data and scaler...")
    joblib.dump((X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test), os.path.join(output_dir, 'processed_data.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print("Data and scaler saved successfully.")

def main():
    """Main function to execute the preprocessing pipeline."""
    data = load_data(DATA_PATH)
    data = clean_data(data)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(data)
    save_data(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, OUTPUT_DIR)
    print("Basic preprocessing pipeline completed successfully.")

if __name__ == "__main__":
    main()
