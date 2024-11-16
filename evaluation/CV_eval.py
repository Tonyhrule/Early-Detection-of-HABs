import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, RepeatedKFold

# Directories
output_dir = "output"
models_dir = "models"
figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)

# Function to perform cross-validation and return results
def cross_validate_model(synthetic):
    """Cross-validate the model and calculate MSE for synthetic or non-synthetic data."""
    # Determine paths based on synthetic flag
    model_path = os.path.join(models_dir, "model_with_synthetic.pkl" if synthetic else "model_non_synthetic.pkl")
    data_path = os.path.join(output_dir, "processed_data_with_synthetic.pkl" if synthetic else "processed_data.pkl")

    # Load model and data
    model = joblib.load(model_path)
    X_train, X_test, y_train, y_test = joblib.load(data_path)

    # Remove NaN values from y_train and corresponding entries in X_train
    mask = ~np.isnan(y_train)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # Cross-validation setup
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
    
    # Display progress
    print(f"Running cross-validation for {'synthetic' if synthetic else 'non-synthetic'} data...")

    # Perform cross-validation and calculate MSE scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    mse_scores = -cv_scores  # Convert to positive MSE
    mean_mse = mse_scores.mean()
    print(f"Mean MSE for {'synthetic' if synthetic else 'non-synthetic'} data: {mean_mse:.4f}")

    return mse_scores, mean_mse

# Run cross-validation for both synthetic and non-synthetic data
mse_scores_non_synthetic, mean_mse_non_synthetic = cross_validate_model(synthetic=False)
mse_scores_synthetic, mean_mse_synthetic = cross_validate_model(synthetic=True)

# Plot cross-validation results for both synthetic and non-synthetic data
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(mse_scores_non_synthetic) + 1), mse_scores_non_synthetic, label="Non-Synthetic Data", marker='o', linestyle='-', color='blue', alpha=0.8)
plt.plot(range(1, len(mse_scores_synthetic) + 1), mse_scores_synthetic, label="Synthetic Data", marker='o', linestyle='-', color='orange', alpha=0.8)

# Labels and styling
plt.xlabel("Fold")
plt.ylabel("Mean Squared Error")
plt.title("Cross-Validation Results for Synthetic and Non-Synthetic Models")
plt.legend()
plt.grid(alpha=0.3)

# Save the plot as PNG
output_file = os.path.join(figures_dir, "cross_validation_results_solid_lines.png")
plt.savefig(output_file, dpi=300)
print(f"Plot saved as {output_file}")

# Show the plot
plt.show()

# Print final scores in the terminal
print("\nFinal MSE Scores:")
print(f"Non-Synthetic Data Mean MSE: {mean_mse_non_synthetic:.4f}")
print(f"Synthetic Data Mean MSE: {mean_mse_synthetic:.4f}")
