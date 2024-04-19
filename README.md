# Fraud Detection Pipeline

This project consists of a pipeline for detecting fraud in bank accounts using machine learning. It performs tasks from data preprocessing to hyperparameter tuning and training the final model.

## Project Contents

The project contains the following files and folders:

- `src/`: Folder containing the source code of the pipeline.
  - `main.py`: The main entry point of the pipeline, which loads data, preprocesses it, tunes hyperparameters, and trains the final model.
- `modules`: Folder containing the pipeline modules.
  - `steps.py`: A module containing the definition of pipeline steps, such as loading data, preprocessing, hyperparameter tuning, and model training.
- `mlproject`: MLflow configuration file.
- `conda_env.yaml`: Conda environment specification file.

## Requirements

- Python 3.x
- MLflow
- Polars
- CatBoost
- Click
- Pandas
- Scikit-learn
- Kaggle
- Optuna

## Installation

1. Clone this repository to your local machine:

```
git clone <REPOSITORY_URL>
cd fraud_detection
```

2. Install the Python dependencies listed in the `conda_env.yaml` file:

```
conda env create -f conda_env.yaml
conda activate fraud_detection
```

3. Configure your Kaggle credentials following the instructions [here](https://www.kaggle.com/docs/api).

## How to Run

1. Start the MLflow server:

```
mlflow ui
```

2. In a new terminal, execute the pipeline using the `mlflow run` command, providing the necessary arguments:

```
mlflow run . --experiment-name fraud -P n_trials=<NUMBER_OF_TRIALS>
```

Replace `<NUMBER_OF_TRIALS>` with the desired number of trials for hyperparameter tuning.

## Notes

- Make sure to have configured your Kaggle credentials correctly before running the pipeline to download the data.
- The training results of the model and tuned hyperparameters will be logged in MLflow.

That's it! You're all set to run the fraud detection pipeline. If you have any questions or encounter any issues, feel free to open an issue.
