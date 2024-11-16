import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np

DATA_PATH_WITH_SYNTHETIC = "output/processed_data_with_synthetic.pkl"
DATA_PATH_NON_SYNTHETIC = "output/processed_data.pkl"
MODEL_OUTPUT_DIR = "models"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

def load_data(path):
    print(f"Loading data from {path}...")
    data = joblib.load(path)
    print(f"Loaded data shapes: X_train={data[0].shape}, y_train={data[2].shape}, X_test={data[1].shape}, y_test={data[3].shape}")
    return data

def reshape_target(y):
    if len(y.shape) > 1:
        print("Reshaping target variable to 1D...")
        y = y.ravel()
    return y

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}

def train_and_evaluate(X_train, X_test, y_train, y_test, output_model_path, use_grid_search=False):
    y_train = reshape_target(y_train)
    y_test = reshape_target(y_test)

    gb = GradientBoostingRegressor(random_state=42)

    if use_grid_search:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        }
        print("Performing Grid Search...")
        grid_search = GridSearchCV(
            gb, param_grid, scoring="r2", cv=5, n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        gb = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")

    print("Training GradientBoostingRegressor...")
    gb.fit(X_train, y_train)

    print("Evaluating model on test data...")
    metrics = evaluate_model(gb, X_test, y_test)
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print(f"Saving model to {output_model_path}...")
    joblib.dump(gb, output_model_path)
    print("Model saved successfully.")

    return metrics

def main():
    print("Training with synthetic data...")
    X_train_syn, X_test_syn, y_train_syn, y_test_syn = load_data(DATA_PATH_WITH_SYNTHETIC)[:4]
    synthetic_metrics = train_and_evaluate(
        X_train_syn, X_test_syn, y_train_syn, y_test_syn,
        os.path.join(MODEL_OUTPUT_DIR, "model_with_synthetic.pkl"),
        use_grid_search=True
    )

    print("\nTraining with non-synthetic data...")
    X_train_non, X_test_non, y_train_non, y_test_non = load_data(DATA_PATH_NON_SYNTHETIC)[:4]
    non_synthetic_metrics = train_and_evaluate(
        X_train_non, X_test_non, y_train_non, y_test_non,
        os.path.join(MODEL_OUTPUT_DIR, "model_non_synthetic.pkl"),
        use_grid_search=True
    )

    print("\n--- Final Summary ---")
    print("Metrics for Synthetic Data:")
    for key, value in synthetic_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nMetrics for Non-Synthetic Data:")
    for key, value in non_synthetic_metrics.items():
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()
