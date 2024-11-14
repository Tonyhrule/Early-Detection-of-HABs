import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

output_dir = "output"
models_dir = "models"

# Function to calculate residuals for a model
def calculate_residuals(synthetic):
    # Determine paths based on synthetic flag
    model_path = os.path.join(models_dir, "stacker_synthetic_model.pkl" if synthetic else "stacker_model.pkl")
    data_path = os.path.join(output_dir, "processed_data_with_synthetic.pkl" if synthetic else "processed_data.pkl")

    # Load model and data
    model = joblib.load(model_path)
    _, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(data_path)

    # Apply polynomial transformation
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_test_poly = poly.fit_transform(X_test_scaled)

    # Make predictions and calculate residuals
    y_pred = model.predict(X_test_poly)

    # Remove NaN values from y_test and corresponding entries in y_pred
    mask = ~np.isnan(y_test)
    y_test = y_test[mask]
    y_pred = y_pred[mask]

    residuals = y_test - y_pred
    return y_test, residuals

# Calculate residuals for both synthetic and non-synthetic models
y_test_non_synthetic, residuals_non_synthetic = calculate_residuals(synthetic=False)
y_test_synthetic, residuals_synthetic = calculate_residuals(synthetic=True)

# Plot residuals for both synthetic and non-synthetic data on the same graph
plt.figure(figsize=(10, 6))
plt.scatter(y_test_non_synthetic, residuals_non_synthetic, label="Non-Synthetic Data", alpha=0.5)
plt.scatter(y_test_synthetic, residuals_synthetic, label="Synthetic Data", alpha=0.5)
plt.hlines(0, min(y_test_non_synthetic.min(), y_test_synthetic.min()), max(y_test_non_synthetic.max(), y_test_synthetic.max()), colors="red", linestyles="dashed")
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.title("Residual Plot for Synthetic and Non-Synthetic Data")
plt.legend()
plt.show()
