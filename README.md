# mlops-project
ci-cd / docker


Telco Customer Churn Prediction with MLOps
This project addresses a supervised classification problem aimed at predicting customer churn in the telecommunications domain. An end-to-end MLOps-oriented architecture is implemented to ensure reproducibility, scalability, and operational reliability.
Model and Dataset

Model: Logistic Regression
Dataset: Imbalanced churn dataset (CSV format)
Number of Samples: 7,032
Number of Features: 22
Evaluation Metrics

Model performance is evaluated using metrics suitable for imbalanced classification tasks:
ROC-AUC
PR-AUC
Precision
Recall
F1-score
MLOps Architecture

MLflow: Experiment tracking, model logging, model registry, and versioning
Prefect: Orchestration of the end-to-end pipeline
(Data → Train → Evaluate → Promote)
Docker: Containerization of the training pipeline for environment-independent execution

