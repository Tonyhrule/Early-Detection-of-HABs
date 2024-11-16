import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

output_dir = "output"
models_dir = "models"

def calculate_metrics(model_path, data_path, scaler_path):
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        X_train, X_test, y_train, y_test = joblib.load(data_path)
        print(f"Data loaded successfully from {data_path}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {data_path}")
    except ValueError as e:
        raise ValueError(f"Error unpacking data: {e}")

    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded successfully from {scaler_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    print(f"Predictions completed. y_pred shape: {y_pred.shape}")

    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    return mse, rmse, mae

if __name__ == "__main__":
    for synthetic, label in [(False, "Non-Synthetic"), (True, "Synthetic")]:
        if synthetic:
            model_path = os.path.join(models_dir, "model_with_synthetic.pkl")
            data_path = os.path.join(output_dir, "processed_data_with_synthetic.pkl")
            scaler_path = os.path.join(output_dir, "scaler_with_synthetic.pkl")
        else:
            model_path = os.path.join(models_dir, "model_non_synthetic.pkl")
            data_path = os.path.join(output_dir, "processed_data.pkl")
            scaler_path = os.path.join(output_dir, "scaler.pkl")

        missing_files = []
        for file_path in [model_path, data_path, scaler_path]:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        if missing_files:
            print(f"\nSkipping {label} data: Missing files:")
            for file in missing_files:
                print(f"  - {file}")
            continue

        print(f"\nCalculating metrics for {label} data...")
        try:
            metrics = calculate_metrics(model_path, data_path, scaler_path)
            mse, rmse, mae = metrics

            print(f"\n{label} Data Evaluation Metrics:")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
        except Exception as e:
            print(f"Error calculating metrics for {label} data: {e}")
