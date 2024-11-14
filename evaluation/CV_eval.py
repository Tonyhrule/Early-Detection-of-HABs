import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt

output_dir = "output"
models_dir = "models"

# Function to perform cross-validation and return results
def cross_validate_model(synthetic):
    # Determine paths based on synthetic flag
    model_path = os.path.join(models_dir, "stacker_synthetic_model.pkl" if synthetic else "stacker_model.pkl")
    data_path = os.path.join(output_dir, "processed_data_with_synthetic.pkl" if synthetic else "processed_data.pkl")

    # Load model and data
    model = joblib.load(model_path)
    _, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(data_path)

    # Apply polynomial transformation
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)

    # Remove NaN values from y_train and corresponding entries in X_train_poly
    mask = ~np.isnan(y_train)
    X_train_poly = X_train_poly[mask]
    y_train = y_train[mask]

    # Cross-validation setup
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
    
    # Display progress
    print(f"Running cross-validation for {'synthetic' if synthetic else 'non-synthetic'} data...")

    # Perform cross-validation and calculate MSE scores
    cv_scores = cross_val_score(model, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')
    mse_scores = -cv_scores  # Convert to positive MSE
    mean_mse = mse_scores.mean()
    print(f"Mean MSE for {'synthetic' if synthetic else 'non-synthetic'} data: {mean_mse:.4f}")

    return mse_scores, mean_mse

# Run cross-validation for both synthetic and non-synthetic data
mse_scores_non_synthetic, mean_mse_non_synthetic = cross_validate_model(synthetic=False)
mse_scores_synthetic, mean_mse_synthetic = cross_validate_model(synthetic=True)

# Plot cross-validation results for both synthetic and non-synthetic data on a single graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(mse_scores_non_synthetic) + 1), mse_scores_non_synthetic, label="Non-Synthetic Data", marker='o')
plt.plot(range(1, len(mse_scores_synthetic) + 1), mse_scores_synthetic, label="Synthetic Data", marker='o')
plt.xlabel("Fold")
plt.ylabel("Mean Squared Error")
plt.title("Cross-Validation Results for Synthetic and Non-Synthetic Models")
plt.legend()  # Ensure the legend appears
plt.grid(True)
plt.show()

# Print final scores in the terminal
print("\nFinal MSE Scores:")
print(f"Non-Synthetic Data Mean MSE: {mean_mse_non_synthetic:.4f}")
print(f"Synthetic Data Mean MSE: {mean_mse_synthetic:.4f}")
