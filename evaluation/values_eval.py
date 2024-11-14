import os
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt

output_dir = "output"
models_dir = "models"

# Function to calculate evaluation metrics
def calculate_metrics(synthetic):
    # Determine paths based on synthetic flag
    model_path = os.path.join(models_dir, "stacker_synthetic_model.pkl" if synthetic else "stacker_model.pkl")
    data_path = os.path.join(output_dir, "processed_data_with_synthetic.pkl" if synthetic else "processed_data.pkl")

    # Load model and data
    model = joblib.load(model_path)
    _, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(data_path)

    # Apply polynomial transformation
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_test_poly = poly.fit_transform(X_test_scaled)

    # Make predictions
    y_pred = model.predict(X_test_poly)

    # Remove NaN values from y_test and corresponding entries in y_pred
    mask = ~np.isnan(y_test)
    y_test = y_test[mask]
    y_pred = y_pred[mask]

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate percent errors for average percent error
    percent_errors = []
    for actual, predicted in zip(y_test, y_pred):
        if actual != 0:
            percent_error = abs((predicted - actual) / actual) * 100
            percent_errors.append(percent_error)

    avg_percent_error = np.mean(percent_errors)

    return mse, rmse, mae, r2, avg_percent_error

# Calculate metrics for both synthetic and non-synthetic data
metrics_non_synthetic = calculate_metrics(synthetic=False)
metrics_synthetic = calculate_metrics(synthetic=True)

# Unpack the metrics for readability
mse_non_synthetic, rmse_non_synthetic, mae_non_synthetic, r2_non_synthetic, avg_percent_error_non_synthetic = metrics_non_synthetic
mse_synthetic, rmse_synthetic, mae_synthetic, r2_synthetic, avg_percent_error_synthetic = metrics_synthetic

# Print the results
print("Evaluation Metrics Summary:")
print("\nNon-Synthetic Data:")
print(f"  MSE: {mse_non_synthetic:.4f}")
print(f"  RMSE: {rmse_non_synthetic:.4f}")
print(f"  MAE: {mae_non_synthetic:.4f}")
print(f"  R² Score: {r2_non_synthetic:.4f}")
print(f"  Average Percent Error: {avg_percent_error_non_synthetic:.2f}%")

print("\nSynthetic Data:")
print(f"  MSE: {mse_synthetic:.4f}")
print(f"  RMSE: {rmse_synthetic:.4f}")
print(f"  MAE: {mae_synthetic:.4f}")
print(f"  R² Score: {r2_synthetic:.4f}")
print(f"  Average Percent Error: {avg_percent_error_synthetic:.2f}%")
