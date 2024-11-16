import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Directories
output_dir = "output"
models_dir = "models"
figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)

def calculate_residuals(model_path, data_path, scaler_path, label):
    """Calculate residuals for a given model and dataset."""
    print(f"Calculating residuals for {label} data...")

    # Load model
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")

    # Load data
    X_train, X_test, y_train, y_test = joblib.load(data_path)
    print(f"Data loaded successfully from {data_path}")

    # Load scaler
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded successfully from {scaler_path}")

    # Scale test data
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    print(f"Predictions completed. y_pred shape: {y_pred.shape}")

    # Calculate residuals
    residuals = y_test - y_pred

    # Remove outliers (top 2% of residuals for better visualization)
    upper_threshold = np.percentile(residuals, 98)
    residuals = residuals[residuals <= upper_threshold]

    return residuals

def plot_residual_distributions(residuals_non_synthetic, residuals_synthetic):
    """Plots residual distributions for synthetic and non-synthetic data."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(residuals_non_synthetic, fill=True, label="Non-Synthetic Data", alpha=0.5)
    sns.kdeplot(residuals_synthetic, fill=True, label="Synthetic Data", alpha=0.5)
    plt.axvline(0, color="red", linestyle="dashed", label="Zero Residual Line")
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.title("Residual Distribution for Synthetic and Non-Synthetic Data")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(figures_dir, "residuals_distribution.png"))
    plt.close()
    print("Residual distribution saved as 'residuals_distribution.png'")

def plot_qq(residuals_non_synthetic, residuals_synthetic):
    """Generates a Q-Q plot comparing synthetic and non-synthetic residuals."""
    plt.figure(figsize=(10, 6))

    # Q-Q Plot for Non-Synthetic Residuals
    probplot_non_synthetic = stats.probplot(residuals_non_synthetic, dist="norm")
    plt.scatter(probplot_non_synthetic[0][0], probplot_non_synthetic[0][1],
                color="blue", alpha=0.3, label="Non-Synthetic Data", s=15)
    plt.plot(probplot_non_synthetic[0][0], probplot_non_synthetic[1][0] * probplot_non_synthetic[0][0] + probplot_non_synthetic[1][1],
             color="darkblue", label="Non-Synthetic Fit", linewidth=1.5, alpha=0.8)

    # Q-Q Plot for Synthetic Residuals
    probplot_synthetic = stats.probplot(residuals_synthetic, dist="norm")
    plt.scatter(probplot_synthetic[0][0], probplot_synthetic[0][1],
                color="orange", alpha=0.3, label="Synthetic Data", s=15)
    plt.plot(probplot_synthetic[0][0], probplot_synthetic[1][0] * probplot_synthetic[0][0] + probplot_synthetic[1][1],
             color="darkorange", label="Synthetic Fit", linewidth=1.5, alpha=0.8)

    # Enhancing visualization
    plt.title("Q-Q Plot for Residuals")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Ordered Residual Quantiles")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(-2.5, 2.5)  # Focus on central residuals
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "qq_plot_residuals_updated.png"))
    plt.close()
    print("Q-Q plot saved as 'qq_plot_residuals_updated.png'")

# Main Execution
if __name__ == "__main__":
    try:
        # Non-Synthetic Data
        model_path_non_synthetic = os.path.join(models_dir, "model_non_synthetic.pkl")
        data_path_non_synthetic = os.path.join(output_dir, "processed_data.pkl")
        scaler_path_non_synthetic = os.path.join(output_dir, "scaler.pkl")
        residuals_non_synthetic = calculate_residuals(
            model_path_non_synthetic, data_path_non_synthetic, scaler_path_non_synthetic, "Non-Synthetic"
        )

        # Synthetic Data
        model_path_synthetic = os.path.join(models_dir, "model_with_synthetic.pkl")
        data_path_synthetic = os.path.join(output_dir, "processed_data_with_synthetic.pkl")
        scaler_path_synthetic = os.path.join(output_dir, "scaler_with_synthetic.pkl")
        residuals_synthetic = calculate_residuals(
            model_path_synthetic, data_path_synthetic, scaler_path_synthetic, "Synthetic"
        )

        # Plot Residual Distribution
        plot_residual_distributions(residuals_non_synthetic, residuals_synthetic)

        # Plot Q-Q Plot
        plot_qq(residuals_non_synthetic, residuals_synthetic)

    except Exception as e:
        print(f"Error: {e}")
