# Synthetic Data Augmentation for Enhancing Harmful Algal Bloom Detection with Machine Learning

Harmful Algal Blooms (HABs) pose significant ecological, economic, and public health challenges. This project investigates the use of synthetic data augmentation, specifically using **Gaussian Copulas**, to enhance machine learning models for early HAB detection. By augmenting real-world datasets with synthetic data, the study aims to address the scarcity of high-quality datasets and improve predictive accuracy for HAB early warning systems.

This repository is an official implementation of **Synthetic Data Augmentation for Enhancing Harmful Algal Bloom Detection with Machine Learning**. It includes all the code and data required to replicate the study's results, which assess the impact of synthetic data volume on model performance.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Repository Structure](#repository-structure)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [License](#license)

---

## Introduction
This project leverages Gaussian Copulas to generate synthetic data that preserves the interdependencies of environmental features such as:
- **Water Temperature (°C)**  
- **Salinity (PSU)**  
- **UVB Radiation (mW/m²)**  

The target variable, **Corrected Chlorophyll-a Concentration (µg/L)**, is a well-established indicator of HAB risk. By systematically analyzing synthetic data volumes ranging from 100 to 1000 samples, the study evaluates their impact on ML model performance.

---

## Installation
Follow the steps below to set up the repository:

### Prerequisites
- Python >= 3.8
- `pip` for package installation
- Recommended: Virtual environment setup using `venv` or `conda`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Tonyhrule/Synthetic-HAB-ML-Augmentation.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   cd hab-detection
   ```

---

## Repository Structure

```bash
├── evaluation/                  # Scripts for evaluating models
│   ├── CV_eval.py               # Cross-validation metrics evaluation
│   ├── percent_error_eval.py    # Percent error evaluation script
│   └── values_eval.py           # General evaluation metrics
├── figures/                     # Contains generated figures for results
├── output/                      # Contains processed data and scaler files
├── Dataset.xlsx                 # Original dataset for preprocessing
├── LICENSE                      # License for the project
├── preprocess_basic.py          # Preprocessing for baseline dataset
├── preprocess_synthetic.py      # Preprocessing for synthetic-augmented dataset
├── README.md                    # Project README
├── requirements.txt             # Python dependencies
└── train.py                     # Script for training the models
```

## Usage

### Data Preprocessing
The `preprocess_basic.py` and `preprocess_synthetic.py` scripts handle preprocessing for the baseline and synthetic datasets, respectively. Run these scripts to clean, scale, and prepare data for training and evaluation.

### Model Training
Use `train.py` to train new models or fine-tune existing ones. The script supports hyperparameter tuning and outputs trained models in the `models/` directory.

### Model Evaluation
Evaluation scripts (`CV_eval.py`, `percent_error_eval.py`, and `values_eval.py`) compute various metrics, including:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Percent Error

---

## Dataset
The original dataset includes environmental features such as:
- Water temperature (°C)
- Salinity (PSU)
- UVB radiation (mW/m²)

**Target Variable**: Corrected chlorophyll-a levels (µg/L).

### Data Preprocessing Steps:
1. Imputation of missing values.
2. Standardization of features.
3. Polynomial feature expansion.

The dataset was sourced from publicly available environmental data and is described in detail in the following publication: [PLoS ONE: Water temperature drives phytoplankton blooms in coastal waters](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0214933#sec020).

Synthetic data was generated using Gaussian Copulas at varying volumes (100, 250, 500, 750, 1000 rows) to analyze its impact on model performance.

---

## Evaluation
Models were evaluated using the following metrics:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **Percent Error**

### Outputs:
- Cross-validation results and error distribution plots are available in the `figures/` directory for more analysis.

#### Cross-Validation Results
![Cross-Validation Results](https://github.com/Tonyhrule/synthetic-hab-ML-augmentation/blob/main/figures/CV_results_final.png)

#### Percent Error Distribution
![Percent Error Distribution](https://github.com/Tonyhrule/synthetic-hab-ML-augmentation/blob/main/figures/Percent_Error_final.png)

---

## Results
Findings from the study include:
- **Improved Accuracy**: Models trained with 100–250 synthetic rows achieved the lowest mean percent error (7.16–7.21%), significantly better than the baseline model (10.17%).
- **Noise from Excessive Synthetic Data**: Models with 1000 synthetic rows showed reduced accuracy due to noise and overfitting.

---

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/Tonyhrule/synthetic-hab-ML-augmentation/blob/main/LICENSE) file for details.
