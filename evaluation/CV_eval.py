import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, RepeatedKFold

output_dir = "output"
base_model_dir = "base model"
synthetic_models_dir = "synthetic models"
data_dir = "data s"
figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)

model_paths = {
    "Base Model": os.path.join(base_model_dir, "model_base.pkl"),
    "Synthetic 100": os.path.join(synthetic_models_dir, "model_synthetic_100.pkl"),
    "Synthetic 250": os.path.join(synthetic_models_dir, "model_synthetic_250.pkl"),
    "Synthetic 500": os.path.join(synthetic_models_dir, "model_synthetic_500.pkl"),
    "Synthetic 750": os.path.join(synthetic_models_dir, "model_synthetic_750.pkl"),
    "Synthetic 1000": os.path.join(synthetic_models_dir, "model_synthetic_1000.pkl"),
}

data_paths = {
    "Base Model": os.path.join(output_dir, "processed_data.pkl"),
    "Synthetic 100": os.path.join(data_dir, "syn_data_100.pkl"),
    "Synthetic 250": os.path.join(data_dir, "syn_data_250.pkl"),
    "Synthetic 500": os.path.join(data_dir, "syn_data_500.pkl"),
    "Synthetic 750": os.path.join(data_dir, "syn_data_750.pkl"),
    "Synthetic 1000": os.path.join(data_dir, "syn_data_1000.pkl"),
}

def cross_validate_model(model_path, data_path):
    model = joblib.load(model_path)
    X_train, X_test, y_train, y_test = joblib.load(data_path)
    mask = ~np.isnan(y_train)
    X_train = X_train[mask]
    y_train = y_train[mask]
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    mse_scores = -cv_scores
    return mse_scores, mse_scores.mean()

results = {}
for label, model_path in model_paths.items():
    data_path = data_paths[label]
    mse_scores, mean_mse = cross_validate_model(model_path, data_path)
    results[label] = (mse_scores, mean_mse)

base_model_mse = results["Base Model"][0]
synthetic_results = {key: results[key][0] for key in results if key != "Base Model"}

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 4]})

ax1.plot(range(1, len(base_model_mse) + 1), base_model_mse, label="Base Model", color="blue", marker="o", linewidth=2)
ax1.set_ylim(0.03, 0.04)
ax1.set_ylabel("MSE (Base Model)", fontsize=14, weight="bold")
ax1.grid(alpha=0.3)

colors = ['orange', 'green', 'red', 'purple', 'brown']
for (label, mse_scores), color in zip(synthetic_results.items(), colors):
    ax2.plot(range(1, len(mse_scores) + 1), mse_scores, label=label, marker="o", linestyle="-", color=color, linewidth=1.5, alpha=0.8)
ax2.set_ylim(0.004, 0.008)
ax2.set_xlabel("Fold", fontsize=14, weight="bold")
ax2.set_ylabel("MSE (Synthetic Models)", fontsize=14, weight="bold")
ax2.grid(alpha=0.3)

for i in range(1, len(base_model_mse) + 1):
    ax1.axvline(i, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.axvline(i, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(labeltop=False)
ax2.tick_params(labelbottom=True)
d = .007
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs) 
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs) 
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs) 

legend = fig.legend(loc="lower left", fontsize=10, title="Models", bbox_to_anchor=(0.1, 0.15), frameon=True, borderpad=1)
legend.get_frame().set_edgecolor("black")

plt.suptitle("Cross-Validation Results Across Models", fontsize=16, weight="bold")
plt.tight_layout(rect=[0, 0.05, 1, 1])
output_file = os.path.join(figures_dir, "cross_validation_results_improved.png")
plt.savefig(output_file, dpi=300)
plt.show()
