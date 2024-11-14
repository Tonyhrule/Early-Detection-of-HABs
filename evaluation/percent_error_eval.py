import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures

output_dir = "output"
models_dir = "models"

# Function to calculate percent error accurately
def calculate_percent_error(synthetic):
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

    # Calculate percent errors
    percent_errors = []
    for actual, predicted in zip(y_test, y_pred):
        if actual != 0:
            percent_error = abs((predicted - actual) / actual) * 100
            percent_errors.append(percent_error)

    return percent_errors

# Calculate percent errors for both synthetic and non-synthetic models
percent_errors_non_synthetic = calculate_percent_error(synthetic=False)
percent_errors_synthetic = calculate_percent_error(synthetic=True)

# Calculate the average percent error for each
avg_percent_error_non_synthetic = np.mean(percent_errors_non_synthetic)
avg_percent_error_synthetic = np.mean(percent_errors_synthetic)

# Plot density plot starting from the origin
plt.figure(figsize=(12, 8))

# Density plot for both datasets, starting from (0,0)
sns.kdeplot(percent_errors_non_synthetic, label="Non-Synthetic Data", color="blue", fill=True, alpha=0.3, linestyle="--")
sns.kdeplot(percent_errors_synthetic, label="Synthetic Data", color="orange", fill=True, alpha=0.3, linestyle="-")

# Annotate mean percent error near the peak for each dataset
plt.text(avg_percent_error_non_synthetic, 0.06, f"Avg Non-Synthetic Error: {avg_percent_error_non_synthetic:.2f}%", color="blue")
plt.text(avg_percent_error_synthetic, 0.05, f"Avg Synthetic Error: {avg_percent_error_synthetic:.2f}%", color="orange")

# Extend x-axis to display the full range up to 100%
plt.xlim(0, 100)
plt.ylim(0, None)  # Let y-axis scale naturally but start at 0

plt.xlabel("Percent Error")
plt.ylabel("Density")
plt.title("Density Plot of Percent Error for Synthetic and Non-Synthetic Data (Starting from Origin)")
plt.legend()
plt.grid(True)
plt.show()

# Print summary metrics in the terminal
print("\nSummary of Percent Error:")
print(f"Average Percent Error for Non-Synthetic Data: {avg_percent_error_non_synthetic:.2f}%")
print(f"Average Percent Error for Synthetic Data: {avg_percent_error_synthetic:.2f}%")
