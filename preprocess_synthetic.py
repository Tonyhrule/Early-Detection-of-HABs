import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from copulas.multivariate import GaussianMultivariate
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "Dataset.xlsx"
OUTPUT_DIR = "output"
SYNTHETIC_DATA_ROWS = 250
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURES = ["Temperature", "Salinity", "UVB"]
TARGET = "ChlorophyllaFlor"

def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    return pd.read_excel(data_path)

def validate_structure(df):
    required_columns = FEATURES + [TARGET]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame does not contain required columns: {required_columns}")
    print("Dataset structure validated.")

def clean_data(data):
    original_shape = data.shape
    data = data.dropna()
    print(f"Original shape: {original_shape}, Cleaned shape: {data.shape}")
    return data

def transform_target(data):
    print("Applying log transformation to the target variable...")
    data[TARGET] = np.log1p(data[TARGET])
    return data

def generate_conditional_synthetic_data(data, num_samples):
    print("Generating synthetic data using Gaussian Copula...")
    model = GaussianMultivariate()
    model.fit(data)
    synthetic_data = model.sample(num_samples)
    synthetic_data = pd.DataFrame(synthetic_data, columns=data.columns)
    synthetic_data = clip_synthetic_data(synthetic_data, data)
    print("Synthetic data generated successfully.")
    return synthetic_data

def clip_synthetic_data(synthetic_data, real_data):
    for column in synthetic_data.columns:
        synthetic_data[column] = synthetic_data[column].clip(real_data[column].min(), real_data[column].max())
    return synthetic_data

def perform_ks_test(data, synthetic_data):
    print("\nPerforming KS test...")
    for column in FEATURES + [TARGET]:
        stat, p_value = ks_2samp(data[column], synthetic_data[column])
        print(f"{column}: KS Statistic = {stat}, p-value = {p_value}")

def preprocess_and_combine(real_data, synthetic_data, weight=0.20):
    print("Combining real and synthetic data...")
    synthetic_data_sampled = synthetic_data.sample(frac=weight, random_state=RANDOM_STATE, replace=True)
    combined_data = pd.concat([real_data, synthetic_data_sampled], ignore_index=True)

    X = combined_data[FEATURES]
    y = combined_data[TARGET]

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def reverse_log_transformation(y_test, y_pred):
    y_test_reversed = np.expm1(y_test)
    y_pred_reversed = np.expm1(y_pred)
    return y_test_reversed, y_pred_reversed

def evaluate_model_performance(y_test, y_pred):
    y_test_reversed, y_pred_reversed = reverse_log_transformation(y_test, y_pred)
    mse = mean_squared_error(y_test_reversed, y_pred_reversed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_reversed, y_pred_reversed)
    r2 = r2_score(y_test_reversed, y_pred_reversed)
    print(f"  MSE: {mse}")
    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")
    print(f"  R² Score: {r2}")

def train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test):
    print("\nTraining Ridge Regression model...")
    ridge_model = Ridge(alpha=10.0)
    ridge_model.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    print("Ridge Regression Performance:")
    evaluate_model_performance(y_test, y_pred_ridge)

    print("\nTraining Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=600, max_depth=30, min_samples_split=4, random_state=RANDOM_STATE)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    print("Random Forest Performance:")
    evaluate_model_performance(y_test, y_pred_rf)

def save_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), os.path.join(output_dir, "processed_data_with_synthetic.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler_with_synthetic.pkl"))

def main():
    data = load_data(DATA_PATH)
    validate_structure(data)

    data_cleaned = clean_data(data)
    data_transformed = transform_target(data_cleaned)

    synthetic_data = generate_conditional_synthetic_data(data_transformed, SYNTHETIC_DATA_ROWS)

    perform_ks_test(data_transformed, synthetic_data)

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_and_combine(data_transformed, synthetic_data, weight=0.20)

    train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)

    save_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler, OUTPUT_DIR)

if __name__ == "__main__":
    main()
