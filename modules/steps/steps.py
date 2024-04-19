import os
import zipfile
import os
import polars as pl
import catboost as cb
import click
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import mlflow
import kaggle
import optuna



class Steps:
    def __init__(self):
        self.TARGET = "fraud_bool"

        self.CATEGORICAL_FEATURES = [
            "payment_type",
            "employment_status",
            "housing_status",
            "source",
            "device_os",
        ]
        self.NUMERICAL_FEATURES = [
            "income",
            "name_email_similarity",
            "prev_address_months_count",
            "current_address_months_count",
            "customer_age",
            "days_since_request",
            "intended_balcon_amount",
            "zip_count_4w",
            "velocity_6h",
            "velocity_24h",
            "velocity_4w",
            "bank_branch_count_8w",
            "date_of_birth_distinct_emails_4w",
            "credit_risk_score",
            "email_is_free",
            "phone_home_valid",
            "phone_mobile_valid",
            "bank_months_count",
            "has_other_cards",
            "proposed_credit_limit",
            "foreign_request",
            "session_length_in_minutes",
            "keep_alive_session",
            "device_distinct_emails_8w",
            "month",
        ]
    def load_raw_data(self, dset_name):
        zip_destination_folder = "./data/"
        raw_destination_folder = os.path.join(zip_destination_folder, "raw")

        # Check if the Kaggle API key was created
        if not os.path.exists(os.path.expanduser(r"C:\\Users\\grego\\.kaggle\\kaggle.json")):
            raise Exception(
                "Kaggle API key not found. Make sure to follow the instructions to set up your Kaggle API key."
            )

        # Download the dataset into a current folder
        kaggle.api.dataset_download_files(
            dset_name,
            path=zip_destination_folder,
        )

        # Check if the destination folder exists, and create it if it does not
        if not os.path.exists(raw_destination_folder):
            os.makedirs(raw_destination_folder)

        # Open the zip file in read mode
        zip_name = os.path.join(
            zip_destination_folder, "bank-account-fraud-dataset-neurips-2022.zip"
        )
        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            # Extract all the files to the destination folder
            zip_ref.extractall(raw_destination_folder)

        # TODO: make file name a param as well
        csv_location = os.path.join(raw_destination_folder, "Base.csv")

        return csv_location


    def process_nans(self, df: pl.DataFrame, drop_thr: float = 0.95) -> pl.DataFrame:
        for col in df.get_columns():
            nulls_prop = col.is_null().mean()
            print(f"{col.name} - {nulls_prop * 100}% missing")
            # drop if missing more than a threshold
            if nulls_prop >= drop_thr:
                print("Dropping", col.name)
                df = df.select([pl.exclude(col.name)])
            # If some values are missing
            elif nulls_prop > 0:
                print("Imputing", col.name)
                # If numeric, impute with median
                if col.is_numeric():
                    fill_value = col.median()
                else:
                    # Else, impute with mode
                    fill_value = col.mode()
                df = df.select(
                    [
                        # Exclude the original column
                        pl.exclude(col.name),
                        # Include the imputed one
                        pl.col(col.name).fill_null(value=fill_value),
                    ]
                )

        return df

    # def drop_static(self, df:pl.DataFrame) -> pl.DataFrame:
    #     for col in df.get_columns():
    #         std = col.std()
    #         # drop if missing more than a threshold
    #         if std == 0:
    #             print("Dropping", col.name)
    #             df = df.select([pl.exclude(col.name)])
        
    #     return df

    def drop_static(self, df: pl.DataFrame) -> pl.DataFrame:
        for col in df.get_columns():
            try:
                std = col.std()
                # drop if missing more than a threshold
                if std == 0:
                    print("Dropping", col.name)
                    df = df.select([pl.exclude(col.name)])
            except pl.PolarsError:
                print(f"Skipping column {col.name} as it contains non-numeric data.")
                continue
        
        return df




    def train_val_test_split(self, df, test_size=0.2, val_size=0.2):
        df_train = df.filter(
            pl.col("month") < df['month'].quantile(0.8)
        )

        df_test = df.filter(
            pl.col("month") >= df['month'].quantile(0.8)
        )

        df_val = df_train.filter(
            pl.col("month") >= df_train['month'].quantile(0.8)
        )

        df_train = df_train.filter(
            pl.col("month") < df_train['month'].quantile(0.8)
        )

        return df_train, df_val, df_test

    def preprocess_data(self, dset_path, missing_thr):
        df = pl.read_csv(dset_path)
        # Preprocess nulls
        df = self.process_nans(df, missing_thr)
        # Drop static
        df = self.drop_static(df)
        # Train/val/test split 
        train_df, val_df, test_df = self.train_val_test_split(df)
        # Save data
        split_destination_folder = './data/processed'
        if not os.path.exists(split_destination_folder):
            os.makedirs(split_destination_folder)

        train_df.write_parquet('./data/processed/train.parquet')
        val_df.write_parquet('./data/processed/validation.parquet')
        test_df.write_parquet('./data/processed/test.parquet')

        file_locations = {
            'train-data-dir': './data/processed/train.parquet',
            'val-data-dir': './data/processed/validation.parquet',
            'test-data-dir': './data/processed/test.parquet',
        }

        return file_locations
    
    def train_model(self, params, train_path, val_path, test_path):
        train_dataset = self.read_cb_data(
            train_path, 
            numeric_features=self.NUMERICAL_FEATURES, 
            categorical_features=self.CATEGORICAL_FEATURES, 
            target_feature=self.TARGET
        )
        val_dataset = self.read_cb_data(
            val_path, 
            numeric_features=self.NUMERICAL_FEATURES, 
            categorical_features=self.CATEGORICAL_FEATURES, 
            target_feature=self.TARGET
        )
        test_dataset = self.read_cb_data(
            test_path, 
            numeric_features=self.NUMERICAL_FEATURES, 
            categorical_features=self.CATEGORICAL_FEATURES, 
            target_feature=self.TARGET
        )
        mlflow.set_experiment("fraud")
        experiment = mlflow.get_experiment_by_name("fraud")
        client = mlflow.tracking.MlflowClient()
        run = client.create_run(experiment.experiment_id)
        with mlflow.start_run(run_id = run.info.run_id):
            gbm = cb.CatBoostClassifier(**params)
            gbm.fit(train_dataset, eval_set=val_dataset, early_stopping_rounds=50)
            preds = gbm.predict_proba(test_dataset)
            ap = average_precision_score(test_dataset.get_label(), preds[:, 1])
            roc = roc_auc_score(test_dataset.get_label(), preds[:, 1])

            mlflow.log_metric("Test ROC AUC", roc)
            mlflow.log_metric("Test PR AUC", ap)
            mlflow.log_params(params)
            mlflow.catboost.log_model(gbm, "catboost_model")
        
        return roc, ap
    
    def read_cb_data(self, 
                     path: str, 
                     numeric_features: list, 
                     categorical_features: list, 
                     target_feature: str
                    ):
        
        data = pd.read_parquet(path)
        dataset = cb.Pool(
            data=data[numeric_features + categorical_features],
            label=data[target_feature],
            cat_features=categorical_features,
        )
        return dataset


    def tune_model(self, train_path, val_path, n_trials):
        train_dataset = self.read_cb_data(
            train_path,
            numeric_features=self.NUMERICAL_FEATURES,
            categorical_features=self.CATEGORICAL_FEATURES,
            target_feature=self.TARGET,
        )
        val_dataset = self.read_cb_data(
            val_path,
            numeric_features=self.NUMERICAL_FEATURES,
            categorical_features=self.CATEGORICAL_FEATURES,
            target_feature=self.TARGET,
        )

        def objective(trial):
            mlflow.set_experiment("fraud")
            experiment = mlflow.get_experiment_by_name("fraud")
            client = mlflow.tracking.MlflowClient()
            run = client.create_run(experiment.experiment_id)
            with mlflow.start_run(run_id = run.info.run_id):
                param = {
                    "n_estimators": 1000,
                    "objective": "Logloss",
                    "subsample": trial.suggest_uniform("subsample", 0.4, 1.0),
                    "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 10.0),
                    "learning_rate": trial.suggest_uniform("learning_rate", 0.006, 0.02),
                    "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 0.5),
                    "depth": trial.suggest_int("depth", 2, 12),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 300),
                }
                mlflow.log_params(param)
                gbm = cb.CatBoostClassifier(**param)
                gbm.fit(train_dataset, eval_set=val_dataset, early_stopping_rounds=50)

                preds = gbm.predict_proba(val_dataset)
                ap = average_precision_score(val_dataset.get_label(), preds[:, 1])
                roc = roc_auc_score(val_dataset.get_label(), preds[:, 1])
                mlflow.log_metric("Val PR AUC", ap)
                mlflow.log_metric("Val ROC AUC", roc)
                return ap

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_trial.params