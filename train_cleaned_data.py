"""
Cleaned Kaggle Data Training Script
====================================
Cleaned Telco Churn datası ile uçtan uca model eğitimi

Usage:
    python train_cleaned_data.py --config config/cleaned_data_config.yaml
"""

import os
import yaml
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ML
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# Imbalance
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# ============================================================
# TRAINER CLASS
# ============================================================

class CleanedDataTrainer:
    """
    Cleaned Kaggle Telco Churn datasıyla model eğitimi
    """

    def __init__(self, config_path=None):
        """Initialize trainer with config"""

        project_root = Path(__file__).parent

        if config_path is None:
            config_path = project_root / "config" / "cleaned_data_config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        if self.config is None:
            raise ValueError(
                f"Config loaded as None. Check encoding/content of {config_path}"
            )

        self.random_state = self.config.get("random_state", 42)
        self.experiment_name = self.config.get(
            "experiment_name", "telco-churn-cleaned-data"
        )

        # MLflow setup
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient()

        print(f"✅ Config loaded from: {config_path}")
        print(f"🧪 Experiment: {self.experiment_name}")

    # --------------------------------------------------------

    def load_data(self):
        """Load cleaned CSV"""
        print("\n📥 Loading data...")

        data_path = self.config["data"]["source"]
        df = pd.read_csv(data_path)

        if df["Churn"].dtype == object:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

        df = df.rename(columns={"Churn": "churn"})

        for col, new_col in {
            "Contract": "contract_type",
            "PaymentMethod": "payment_method"
        }.items():
            if col in df.columns:
                df = df.rename(columns={col: new_col})

        for c in ["service_combo_id", "geo_code"]:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        churn_rate = df["churn"].mean()

        print(f"   Shape: {df.shape}")
        print(f"   Churn rate: {churn_rate:.2%}")

        return df

    # --------------------------------------------------------

    def create_feature_crosses(self, df):
        print("\n🔀 Creating feature crosses...")

        def cross(a, b):
            return a.astype(str) + "__x__" + b.astype(str)

        df["cross_contract_payment"] = cross(
            df["contract_type"], df["payment_method"]
        )
        df["cross_service_contract"] = cross(
            df["service_combo_id"], df["contract_type"]
        )
        df["cross_geo_contract"] = cross(
            df["geo_code"], df["contract_type"]
        )

        return df

    # --------------------------------------------------------

    def prepare_features(self, df):
        print("\n📊 Preparing features...")

        y = df["churn"].astype(int)
        X = df.drop(columns=["churn"])

        numeric_features = ["tenure", "MonthlyCharges"]
        if "TotalCharges" in X.columns:
            numeric_features.append("TotalCharges")

        categorical_features = [
            c for c in [
                "contract_type", "payment_method", "gender",
                "Partner", "Dependents", "PaperlessBilling",
                "SeniorCitizen"
            ] if c in X.columns
        ]

        service_cols = [
            "InternetService", "PhoneService", "MultipleLines",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        categorical_features += [c for c in service_cols if c in X.columns]

        cross_features = [
            "cross_contract_payment",
            "cross_service_contract",
            "cross_geo_contract"
        ]

        high_card_features = ["service_combo_id", "geo_code"]

        return X, y, numeric_features, categorical_features, cross_features, high_card_features

    # --------------------------------------------------------

    def build_pipeline(
        self,
        numeric_features,
        categorical_features,
        cross_features,
        high_card_features,
        model_type="baseline"
    ):
        print(f"\n🏗️ Building pipeline: {model_type}")

        class HashingTransformer(BaseEstimator, TransformerMixin):
            def __init__(self, cols, n_features=2**18):
                self.cols = cols
                self.n_features = n_features

            def fit(self, X, y=None):
                self.hasher_ = FeatureHasher(
                    n_features=self.n_features,
                    input_type="string"
                )
                return self

            def transform(self, X):
                tokens = []
                for _, row in X[self.cols].iterrows():
                    tokens.append([f"{c}={row[c]}" for c in self.cols])
                return self.hasher_.transform(tokens)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"),
                 categorical_features + cross_features),
                ("hash", HashingTransformer(high_card_features),
                 high_card_features),
            ],
            sparse_threshold=0.3
        )

        if model_type == "baseline":
            model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=self.random_state
            )
        else:
            model = RandomForestClassifier(
                n_estimators=400,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=self.random_state
            )

        sampler = RandomOverSampler(random_state=self.random_state)

        pipeline = ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("sampler", sampler),
            ("model", model)
        ])

        return pipeline

    # --------------------------------------------------------

    def evaluate(self, name, pipeline, X_train, X_test, y_train, y_test):
        print(f"\n🎯 Training: {name}")

        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics = {
            "pr_auc": average_precision_score(y_test, proba),
            "roc_auc": roc_auc_score(y_test, proba),
            "f1": f1_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "cm": confusion_matrix(y_test, pred)
        }

        print(classification_report(y_test, pred, digits=3))
        return metrics, pipeline

    # --------------------------------------------------------

    def run(self):
        print("\n🚀 STARTING TRAINING PIPELINE")

        df = self.load_data()
        df = self.create_feature_crosses(df)

        X, y, num, cat, cross, high = self.prepare_features(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config["data"].get("test_size", 0.2),
            stratify=y,
            random_state=self.random_state
        )

        pipeline = self.build_pipeline(num, cat, cross, high)
        metrics, pipeline = self.evaluate(
            "Baseline LogisticRegression",
            pipeline,
            X_train, X_test,
            y_train, y_test
        )

        print("\n✅ TRAINING FINISHED")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/cleaned_data_config.yaml",
        help="Config file path"
    )
    args = parser.parse_args()

    trainer = CleanedDataTrainer(config_path=args.config)
    trainer.run()
