# IPL Win Predictor - MLOps Project Report

## 1. Project Overview
This project implements a production-grade MLOps pipeline for predicting the winner of IPL cricket matches. The system automates the entire machine learning lifecycle, from data ingestion to deployment and monitoring, ensuring reproducibility, scalability, and reliability.

## 2. Architecture & Technologies

### 2.1 Core Components
- **Data Processing**: Apache Spark (via PySpark) is used for distributed data processing and feature engineering, handling large datasets efficiently.
- **Model Training**: XGBoost classifier trained on historical match data.
- **Orchestration**: Apache Airflow manages the workflow DAG (Directed Acyclic Graph), ensuring tasks run in the correct order and handling retries/failures.
- **Data & Model Versioning**:
    - **DVC (Data Version Control)** tracks data files, models, and pipeline stages (`dvc.yaml`), enabling reproducibility.
    - **MLflow** tracks experiments (hyperparameters, metrics) and manages the Model Registry for versioning trained models.
- **Serving**: FastAPI provides a RESTful API for real-time predictions.
- **Containerization**: Docker and Docker Compose containerize all services, ensuring consistent environments across development and production.

### 2.2 DevOps & Monitoring
- **CI/CD**: GitHub Actions automates testing, linting, Docker image building, and deployment.
- **Monitoring Stack**:
    - **Prometheus**: Collects system and application metrics (e.g., API latency, prediction counts).
    - **Grafana**: Visualizes metrics on interactive dashboards.
    - **AlertManager**: Handles alerting rules (e.g., high API error rates).

## 3. Pipeline Implementation

### 3.1 ML Pipeline (Airflow + DVC)
The Airflow DAG (`ipl_ml_pipeline`) orchestrates the following steps:
1. **Data Quality Check**: Validates raw data existence and basic quality checks.
2. **Data Processing**: `dvc repro data_processing` - Cleans raw data and converts it to Parquet format.
3. **Feature Engineering**: `dvc repro feature_engineering` - Calculates derived features like team win percentages and head-to-head stats.
4. **Model Training**: `dvc repro model_training` - Trains the XGBoost model.
    - Logs parameters and metrics to MLflow.
    - Registers the model to the MLflow Model Registry.
    - Transitions the model to "Staging".
5. **Model Evaluation**: `dvc repro model_evaluation` - Generates performance reports (confusion matrix, ROC curve).
6. **Deployment**: Deploys the trained model artifacts to the serving directory.

### 3.2 Model Serving
The FastAPI service (`model-api`) loads the trained model and scaler. It exposes endpoints for:
- `/predict`: Real-time match prediction.
- `/health`: Service health status.
- `/metrics`: Prometheus metrics.
- `/model-info`: Information about the currently loaded model.

## 4. Key Features

- **Automatic Retraining**: Modifying `configs/model_config.yaml` or data files triggers DVC to detect changes. The Airflow DAG automatically retrains the model only when dependencies change.
- **Experiment Tracking**: Every training run is logged in MLflow, allowing comparison of different hyperparameter configurations.
- **Robust Error Handling**: The pipeline includes data quality checks and fail-safes to prevent bad models from being deployed.
- **Scalable Deployment**: Dockerized services can be deployed to any container orchestration platform (e.g., Kubernetes, ECS).

## 5. Setup & Usage Summary

The entire stack is brought up using `docker compose up -d`.
- **Airflow** (port 8081) is the control center for running the pipeline.
- **MLflow** (port 5000) provides visibility into training results and model versions.
- **Grafana** (port 3001) offers real-time monitoring of the system's health.

## 6. Future Improvements
- **Model Drift Detection**: Implement drift detection to trigger retraining automatically when data distribution shifts.
- **A/B Testing**: Deploy multiple model versions simultaneously to compare performance in production.
- **Kubernetes Deployment**: Migrate from Docker Compose to Kubernetes for better scalability and fault tolerance.

