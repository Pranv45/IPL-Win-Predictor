# IPL Win Predictor MLOps Pipeline

An end-to-end MLOps system for predicting IPL match winners, featuring automated training, deployment, and monitoring.

## 🏗 Architecture

- **Orchestration**: Apache Airflow
- **Data Versioning**: DVC (Data Version Control)
- **Experiment Tracking**: MLflow
- **Model Serving**: FastAPI
- **Monitoring**: Prometheus & Grafana
- **Containerization**: Docker & Docker Compose
- **CI/CD**: GitHub Actions

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/IPL-Win-Predictor.git
   cd IPL-Win-Predictor
   ```

2. Start all services:
   ```bash
   docker compose up -d --build
   ```

3. Access the services:
   | Service | URL | Credentials |
   |---------|-----|-------------|
   | **Airflow** | http://localhost:8081 | `admin` / `admin` |
   | **MLflow** | http://localhost:5000 | - |
   | **Prediction API** | http://localhost:8080 | - |
   | **Grafana** | http://localhost:3001 | `admin` / `admin` |
   | **Prometheus** | http://localhost:9090 | - |
   | **Frontend** | http://localhost:3002 | - |

## 🔄 Pipeline Workflow

The ML pipeline is defined in Airflow (`dags/ml_pipeline_dag.py`) and orchestrated via DVC:

1. **Data Processing**: Ingests and cleans raw IPL data using Spark.
2. **Feature Engineering**: Creates features like win percentages and recent form.
3. **Model Training**: Trains an XGBoost model and logs metrics/artifacts to MLflow.
4. **Evaluation**: Validates model performance against thresholds.
5. **Deployment**: Registers the model to MLflow Registry and deploys it to the API.

To trigger the pipeline manually:
1. Go to Airflow UI (http://localhost:8081)
2. Enable `ipl_ml_pipeline` DAG
3. Click the "Trigger DAG" (Play) button

## 🛠 Configuration

- **Hyperparameters**: Edit `configs/model_config.yaml` to change model parameters. DVC will automatically detect changes and retrain.
- **Monitoring**: Alerts and dashboards are configured in `monitoring/`.

## 📦 CI/CD

The GitHub Actions pipeline (`.github/workflows/ci-cd.yml`) performs:
1. **Test**: Linting and unit tests.
2. **Validate**: Checks model performance metrics.
3. **Build**: Builds and pushes Docker images.
4. **Deploy**: Deploys to remote server via SSH (if secrets are configured).

## 📝 Usage

### API Prediction
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "team1": "Mumbai Indians",
    "team2": "Chennai Super Kings",
    "venue": "Wankhede Stadium",
    "city": "Mumbai",
    "toss_winner": "Mumbai Indians",
    "toss_decision": "field"
  }'
```

### Retraining
To retrain with new hyperparameters:
1. Modify `configs/model_config.yaml`
2. Trigger the DAG in Airflow
3. Check MLflow for the new model version

---
**Note**: Before pushing to Git, update `YOUR_USERNAME` in the clone command above and ensure `dvc.yaml` and `docker-compose.yml` are committed.

