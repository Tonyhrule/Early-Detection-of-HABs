import os
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(output_dir, synthetic=False):
    data_file = 'processed_data_with_synthetic.pkl' if synthetic else 'processed_data.pkl'
    logging.info(f"Loading data from {data_file}")
    return joblib.load(os.path.join(output_dir, data_file))

def save_model(model, model_name, models_dir):
    joblib.dump(model, os.path.join(models_dir, f'{model_name}_model.pkl'))
    logging.info(f'{model_name} model saved.')

def hyperparameter_tuning(model, param_grid, X_train, y_train, random_search=True):
    search = (RandomizedSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1, random_state=42)
              if random_search else
              GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1))
    search.fit(X_train, y_train)
    logging.info(f"Best parameters for {model.__class__.__name__}: {search.best_params_}")
    return search.best_estimator_

def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        save_model(model, name, models_dir)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"{name} Mean Squared Error: {mse:.4f}")
    return fitted_models

def plot_cv_results(cv_results, model_names):
    plt.figure(figsize=(10, 6))
    for model_name, results in zip(model_names, cv_results):
        plt.plot(range(1, len(results) + 1), -results, label=model_name, marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Mean Squared Error')
    plt.title('Cross-Validation Results')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_residuals(models, X_test, y_test):
    plt.figure(figsize=(14, 8))
    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        plt.subplot(2, 2, i + 1)
        plt.scatter(y_test, residuals, s=10, alpha=0.7)
        plt.hlines(0, min(y_test), max(y_test), colors='r', linestyles='dashed')
        plt.xlabel('Actual')
        plt.ylabel('Residuals')
        plt.title(f'{name} Residuals')
    plt.tight_layout()
    plt.show()

def preprocess_data(X_train, X_test, y_train, y_test):
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    y_train = imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test = imputer.transform(y_test.values.reshape(-1, 1)).ravel()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    return X_train_poly, X_test_poly, y_train, y_test

output_dir = 'output'
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

synthetic = False  # Set to False for training on base data only
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_data(output_dir, synthetic=synthetic)
X_train_poly, X_test_poly, y_train, y_test = preprocess_data(X_train_scaled, X_test_scaled, y_train, y_test)

# Define models and parameter grids
param_grid_rf = {
    'n_estimators': [50, 100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
param_grid_gb = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6]
}
param_grid_nn = {
    'hidden_layer_sizes': [(50,), (100,), (200,), (100, 100), (200, 200)],
    'learning_rate_init': [0.001, 0.01, 0.05, 0.1],
    'max_iter': [500, 1000]
}

# Train models with hyperparameter tuning
rf = hyperparameter_tuning(RandomForestRegressor(random_state=42), param_grid_rf, X_train_poly, y_train)
gb = hyperparameter_tuning(GradientBoostingRegressor(random_state=42), param_grid_gb, X_train_poly, y_train)
nn = hyperparameter_tuning(MLPRegressor(random_state=42, early_stopping=True), param_grid_nn, X_train_poly, y_train)

# Ensemble model setup
estimators = [('rf', rf), ('gb', gb), ('nn', nn)]
stacker = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stacker.fit(X_train_poly, y_train)
save_model(stacker, 'stacker', models_dir)

# Cross-validation scores
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
cv_results_rf = cross_val_score(rf, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')
cv_results_gb = cross_val_score(gb, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')
cv_results_nn = cross_val_score(nn, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')
cv_results_stacker = cross_val_score(stacker, X_train_poly, y_train, cv=cv, scoring='neg_mean_squared_error')

plot_cv_results([cv_results_rf, cv_results_gb, cv_results_nn, cv_results_stacker], 
                ['Random Forest', 'Gradient Boosting', 'Neural Network', 'Stacked Regressor'])

# Evaluate models on test data
models = {'Random Forest': rf, 'Gradient Boosting': gb, 'Neural Network': nn, 'Ensemble': stacker}
plot_residuals(models, X_test_poly, y_test)
