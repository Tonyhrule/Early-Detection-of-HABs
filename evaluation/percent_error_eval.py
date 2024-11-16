import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = "output"
base_model_dir = "base model"
synthetic_models_dir = "synthetic models"
figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)

synthetic_levels = [100, 250, 500, 750, 1000]
MAX_PERCENT_ERROR = 50

def calculate_percent_error(model_path, data_path):
    model = joblib.load(model_path)
    X_train, X_test, y_train, y_test = joblib.load(data_path)
    y_pred = model.predict(X_test)
    mask = ~np.isnan(y_test)
    y_test = y_test[mask]
    y_pred = y_pred[mask]
    print(f"\nStatistics for {os.path.basename(model_path)}:")
    print(f"  y_test: min={y_test.min():.2f}, max={y_test.max():.2f}, mean={y_test.mean():.2f}")
    print(f"  y_pred: min={y_pred.min():.2f}, max={y_pred.max():.2f}, mean={y_pred.mean():.2f}")
    percent_errors = np.abs((y_pred - y_test) / y_test) * 100
    percent_errors = np.clip(percent_errors, 0, MAX_PERCENT_ERROR)
    print(f"  Percent Errors: mean={percent_errors.mean():.2f}%, median={np.median(percent_errors):.2f}%, std={percent_errors.std():.2f}%")
    return percent_errors

percent_errors_data = {}
mean_percent_errors = {}

print("Calculating percent error for the base model...")
base_model_path = os.path.join(base_model_dir, "model_base.pkl")
base_data_path = os.path.join(output_dir, "processed_data.pkl")
percent_errors = calculate_percent_error(base_model_path, base_data_path)
percent_errors_data["Base Model"] = percent_errors
mean_percent_errors["Base Model"] = np.mean(percent_errors)

for level in synthetic_levels:
    print(f"Calculating percent error for synthetic model with {level} rows...")
    model_path = os.path.join(synthetic_models_dir, f"model_synthetic_{level}.pkl")
    data_path = os.path.join(output_dir, "processed_data_with_synthetic.pkl")
    percent_errors = calculate_percent_error(model_path, data_path)
    percent_errors_data[f"Synthetic {level}"] = percent_errors
    mean_percent_errors[f"Synthetic {level}"] = np.mean(percent_errors)

print("\nMean Percent Error for Each Model:")
for label, mean_error in mean_percent_errors.items():
    print(f"{label}: {mean_error:.2f}%")

plt.figure(figsize=(12, 8))
colors = sns.color_palette("Spectral", len(percent_errors_data))
line_width = 2.5

for (label, percent_errors), color in zip(percent_errors_data.items(), colors):
    sns.kdeplot(
        percent_errors,
        label=label,
        color=color,
        fill=True,
        alpha=0.25,
        linewidth=line_width
    )

plt.xlim(0, MAX_PERCENT_ERROR)
plt.ylim(0, None)
plt.xlabel("Percent Error (%)", fontsize=14, weight="bold")
plt.ylabel("Density", fontsize=14, weight="bold")
plt.title("Density Plot of Percent Error Across Models", fontsize=16, weight="bold")
plt.legend(title="Models", fontsize=12, loc="upper right")
plt.grid(alpha=0.3)
plt.tight_layout()

output_file = os.path.join(figures_dir, "percent_error_density_final.png")
plt.savefig(output_file, dpi=300)
print(f"\nPlot saved as {output_file}")

plt.show()
