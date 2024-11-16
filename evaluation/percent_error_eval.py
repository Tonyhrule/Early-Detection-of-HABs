import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Directories
output_dir = "output"
models_dir = "models"
figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)  # Create figures directory if it doesn't exist

# Maximum percent error to include in the plot (for clipping outliers)
MAX_PERCENT_ERROR = 100

def calculate_percent_error(synthetic):
    """Calculate percent error for synthetic or non-synthetic data."""
    # Determine paths based on synthetic flag
    model_path = os.path.join(models_dir, "model_with_synthetic.pkl" if synthetic else "model_non_synthetic.pkl")
    data_path = os.path.join(output_dir, "processed_data_with_synthetic.pkl" if synthetic else "processed_data.pkl")

    # Load model and data
    model = joblib.load(model_path)
    X_train, X_test, y_train, y_test = joblib.load(data_path)

    # Make predictions
    y_pred = model.predict(X_test)

    # Remove NaN values from y_test and corresponding entries in y_pred
    mask = ~np.isnan(y_test)
    y_test = y_test[mask]
    y_pred = y_pred[mask]

    # Calculate percent errors (vectorized)
    percent_errors = np.abs((y_pred - y_test) / y_test) * 100
    percent_errors = np.clip(percent_errors, 0, MAX_PERCENT_ERROR)  # Clip outliers
    return percent_errors

# Calculate percent errors for both datasets
percent_errors_non_synthetic = calculate_percent_error(synthetic=False)
percent_errors_synthetic = calculate_percent_error(synthetic=True)

# Calculate the average percent error for each
avg_percent_error_non_synthetic = np.mean(percent_errors_non_synthetic)
avg_percent_error_synthetic = np.mean(percent_errors_synthetic)

# Plot density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(percent_errors_non_synthetic, label="Non-Synthetic Data", color="blue", fill=True, alpha=0.3, linestyle="--")
sns.kdeplot(percent_errors_synthetic, label="Synthetic Data", color="orange", fill=True, alpha=0.3, linestyle="-")

# Adjust x-axis and y-axis limits
plt.xlim(0, MAX_PERCENT_ERROR)
plt.ylim(0, None)

# Labels, title, and legend
plt.xlabel("Percent Error (%)")
plt.ylabel("Density")
plt.title("Density Plot of Percent Error for Synthetic and Non-Synthetic Data")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save the plot as a PNG
output_file = os.path.join(figures_dir, "percent_error_density.png")
plt.savefig(output_file, dpi=300)
print(f"Plot saved as {output_file}")

print("\nSummary of Percent Error:")
print(f"Average Percent Error for Non-Synthetic Data: {avg_percent_error_non_synthetic:.2f}%")
print(f"Average Percent Error for Synthetic Data: {avg_percent_error_synthetic:.2f}%")

plt.show()

