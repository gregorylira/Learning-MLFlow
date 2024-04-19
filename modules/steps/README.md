## Module README

### Overview

This module provides functionality for processing, training, and evaluating machine learning models for fraud detection using the CatBoost algorithm. It includes various steps such as data preprocessing, model training, hyperparameter tuning, and evaluation.

### Dependencies

- `os`: Operating system dependent functionality.
- `zipfile`: Work with ZIP archives.
- `polars`: Data manipulation library.
- `catboost`: Gradient boosting library for classification.
- `click`: Command-line interface creation kit.
- `pandas`: Data manipulation and analysis library.
- `sklearn`: Machine learning library for Python.
- `mlflow`: Open-source platform for managing the end-to-end machine learning lifecycle.
- `kaggle`: Kaggle API client.
- `optuna`: Hyperparameter optimization framework.

### Usage

1. **Loading Raw Data**: Use the `load_raw_data` method to download and extract the raw dataset from Kaggle. Make sure to set up your Kaggle API key as per the instructions.

   ```python
   steps = Steps()
   csv_location = steps.load_raw_data("dataset_name")
   ```

2. **Preprocessing Data**: Preprocess the raw data by handling missing values, dropping static columns, and splitting the dataset into training, validation, and testing sets.

   ```python
   file_locations = steps.preprocess_data(csv_location, missing_thr=0.95)
   ```

3. **Training Model**: Train a CatBoost classifier using the provided training data and evaluate it on the validation set. The trained model will be logged using MLflow.

   ```python
   params = {...}  # Define CatBoost hyperparameters
   roc_auc, pr_auc = steps.train_model(params, file_locations['train-data-dir'], file_locations['val-data-dir'], file_locations['test-data-dir'])
   ```

4. **Hyperparameter Tuning**: Perform hyperparameter tuning using Optuna to find the optimal set of hyperparameters for the CatBoost classifier.
   ```python
   best_params = steps.tune_model(file_locations['train-data-dir'], file_locations['val-data-dir'], n_trials=100)
   ```

### Class: `Steps`

#### Methods:

- `load_raw_data(dset_name)`: Downloads and extracts the raw dataset from Kaggle.
- `process_nans(df, drop_thr)`: Handles missing values in the DataFrame.
- `drop_static(df)`: Drops static columns from the DataFrame.
- `train_val_test_split(df, test_size, val_size)`: Splits the DataFrame into training, validation, and testing sets.
- `preprocess_data(dset_path, missing_thr)`: Preprocesses the raw data and saves the processed datasets.
- `train_model(params, train_path, val_path, test_path)`: Trains a CatBoost model and evaluates it on the test set.
- `read_cb_data(path, numeric_features, categorical_features, target_feature)`: Reads the data from the specified path and formats it for CatBoost.
- `tune_model(train_path, val_path, n_trials)`: Performs hyperparameter tuning for the CatBoost model.

### Example

```python
import os
from module_name import Steps

# Initialize Steps class
steps = Steps()

# Load raw data
csv_location = steps.load_raw_data("dataset_name")

# Preprocess data
file_locations = steps.preprocess_data(csv_location, missing_thr=0.95)

# Train model
params = {...}  # Define CatBoost hyperparameters
roc_auc, pr_auc = steps.train_model(params, file_locations['train-data-dir'], file_locations['val-data-dir'], file_locations['test-data-dir'])

# Hyperparameter tuning
best_params = steps.tune_model(file_locations['train-data-dir'], file_locations['val-data-dir'], n_trials=100)
```

### Note

- Make sure to replace `"dataset_name"` with the appropriate Kaggle dataset name.
- Customize hyperparameters according to your specific requirements for model training and tuning.
