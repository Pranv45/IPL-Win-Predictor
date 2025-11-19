from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import json
import logging
import os
import yaml
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import start_http_server
import time
from typing import List, Optional
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI(title="IPL Win Predictor API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
PREDICTION_COUNTER = Counter('ipl_predictions_total', 'Total number of predictions made')
PREDICTION_LATENCY = Histogram('ipl_prediction_latency_seconds', 'Prediction latency in seconds')
MODEL_LOAD_TIME = Histogram('ipl_model_load_time_seconds', 'Model loading time in seconds')

# Global variables for model and scaler
model = None
scaler = None
model_info = None
label_encoders = None

class PredictionRequest(BaseModel):
    team1: str
    team2: str
    venue: str
    city: str
    team1_win_percentage: float
    team2_win_percentage: float
    team1_recent_form: Optional[float] = None
    team2_recent_form: Optional[float] = None
    team1_head_to_head: Optional[float] = None
    team2_head_to_head: Optional[float] = None

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    confidence: str
    features_used: List[str]

def load_config():
    """Load model configuration."""
    config_path = 'configs/model_config.yaml'
    if not os.path.exists(config_path):
        logging.warning(f"Config file not found: {config_path}. Using defaults.")
        config = {
            'mlflow': {
                'tracking_uri': 'file:./mlruns',
                'registered_model_name': 'IPLWinPredictor',
                'model_stage': 'Production'
            }
        }
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Override MLflow tracking URI from environment variable if set (for Docker)
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if mlflow_tracking_uri:
        if 'mlflow' not in config:
            config['mlflow'] = {}
        config['mlflow']['tracking_uri'] = mlflow_tracking_uri
        logging.info(f"Using MLflow tracking URI from environment: {mlflow_tracking_uri}")

    return config

def load_model():
    """Load the trained model and artifacts from MLflow Model Registry or file system fallback."""
    global model, scaler, model_info, label_encoders

    start_time = time.time()
    config = load_config()

    try:
        # Try to load from MLflow Model Registry first
        mlflow_config = config.get('mlflow', {})
        registered_model_name = mlflow_config.get('registered_model_name', 'IPLWinPredictor')
        model_stage = mlflow_config.get('model_stage', 'Production')
        tracking_uri = mlflow_config.get('tracking_uri', 'file:./mlruns')

        mlflow.set_tracking_uri(tracking_uri)

        try:
            client = MlflowClient()

            # Get latest version from specified stage
            if model_stage and model_stage != "None":
                latest_versions = client.get_latest_versions(
                    name=registered_model_name,
                    stages=[model_stage]
                )
                if not latest_versions:
                    raise ValueError(f"No model found in {model_stage} stage. Trying Production...")
                model_version = latest_versions[0]
            else:
                # Get latest version regardless of stage
                latest_versions = client.get_latest_versions(name=registered_model_name)
                if not latest_versions:
                    raise ValueError(f"No registered model found: {registered_model_name}")
                model_version = latest_versions[0]

            # Load model from registry
            model_uri = f"models:/{registered_model_name}/{model_stage if model_stage != 'None' else model_version.version}"
            logging.info(f"Loading model from MLflow Registry: {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)

            # Load scaler and model_info from MLflow artifacts
            # Get the run that created this model version
            run_id = model_version.run_id
            artifacts_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="artifacts",
                dst_path=None
            )

            # Load scaler
            scaler_path = os.path.join(artifacts_path, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                raise FileNotFoundError(f"Scaler not found in MLflow artifacts: {scaler_path}")

            # Load model info
            model_info_path = os.path.join(artifacts_path, "model_info.json")
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
            else:
                raise FileNotFoundError(f"Model info not found in MLflow artifacts: {model_info_path}")

            logging.info(f"Model loaded from MLflow Registry - Version: {model_version.version}, Stage: {model_version.current_stage}")

        except Exception as mlflow_error:
            logging.warning(f"Failed to load from MLflow Registry: {mlflow_error}")
            logging.info("Falling back to file system...")

            # Fallback to file system
            model_path = 'models/ipl_win_predictor.pkl'
            scaler_path = 'models/scaler.pkl'
            model_info_path = 'models/model_info.json'

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            if not os.path.exists(model_info_path):
                raise FileNotFoundError(f"Model info file not found: {model_info_path}")

            # Load from file system
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)

            logging.info("Model loaded from file system (fallback)")

        # Load label encoders (always from file system as they're not in MLflow)
        label_encoders_path = 'data/features/label_encoders.pkl'
        if not os.path.exists(label_encoders_path):
            raise FileNotFoundError(f"Label encoders file not found: {label_encoders_path}")
        with open(label_encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)

        load_time = time.time() - start_time
        MODEL_LOAD_TIME.observe(load_time)

        logging.info(f"Model loaded successfully in {load_time:.2f} seconds")
        logging.info(f"Model: {model_info.get('model_name', 'unknown')}, Features: {len(model_info.get('feature_columns', []))}")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def prepare_features(request: PredictionRequest) -> np.ndarray:
    """Prepare features for prediction."""
    if model_info is None or label_encoders is None:
        raise ValueError("Model or label encoders not loaded")

    # Create feature dictionary
    features = {
        'team1_win_percentage': request.team1_win_percentage,
        'team2_win_percentage': request.team2_win_percentage,
        'team1_recent_form': request.team1_recent_form or 0.5,
        'team2_recent_form': request.team2_recent_form or 0.5,
        'team1_head_to_head': request.team1_head_to_head or 0.5,
        'team2_head_to_head': request.team2_head_to_head or 0.5,
    }

    # Encode categorical features
    # label_encoders are dictionaries (not sklearn objects) created in feature_engineering.py
    if 'Team1_encoded' in model_info.get('feature_columns', []):
        features['Team1_encoded'] = label_encoders.get('Team1', {}).get(request.team1, -1)
    if 'Team2_encoded' in model_info.get('feature_columns', []):
        features['Team2_encoded'] = label_encoders.get('Team2', {}).get(request.team2, -1)
    if 'Venue_encoded' in model_info.get('feature_columns', []):
        features['Venue_encoded'] = label_encoders.get('Venue', {}).get(request.venue, -1)
    if 'City_encoded' in model_info.get('feature_columns', []):
        features['City_encoded'] = label_encoders.get('City', {}).get(request.city, -1)

    # Create interaction features
    features['win_percentage_diff'] = request.team1_win_percentage - request.team2_win_percentage
    features['recent_form_diff'] = (request.team1_recent_form or 0.5) - (request.team2_recent_form or 0.5)
    features['h2h_advantage'] = (request.team1_head_to_head or 0.5) - (request.team2_head_to_head or 0.5)

    # Create feature array in the correct order
    feature_array = []
    for feature in model_info['feature_columns']:
        if feature in features:
            feature_array.append(features[feature])
        else:
            feature_array.append(0)  # Default value for missing features

    return np.array(feature_array).reshape(1, -1)

def get_confidence_level(probability: float) -> str:
    """Get confidence level based on probability."""
    if probability >= 0.8 or probability <= 0.2:
        return "High"
    elif probability >= 0.6 or probability <= 0.4:
        return "Medium"
    else:
        return "Low"

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()
    # Start Prometheus metrics server on port 8000
    # This exposes metrics at /metrics endpoint for Prometheus scraping
    start_http_server(8000, addr='0.0.0.0')

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "IPL Win Predictor API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Prepare features
        features = prepare_features(request)

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        # Calculate latency
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_COUNTER.inc()

        # Get confidence level
        confidence = get_confidence_level(probability)

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence,
            features_used=model_info['feature_columns']
        )

    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get model information."""
    if model_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Try to get MLflow registry info
    registry_info = {}
    try:
        config = load_config()
        mlflow_config = config.get('mlflow', {})
        registered_model_name = mlflow_config.get('registered_model_name', 'IPLWinPredictor')
        model_stage = mlflow_config.get('model_stage', 'Production')
        tracking_uri = mlflow_config.get('tracking_uri', 'file:./mlruns')

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        if model_stage and model_stage != "None":
            latest_versions = client.get_latest_versions(
                name=registered_model_name,
                stages=[model_stage]
            )
        else:
            latest_versions = client.get_latest_versions(name=registered_model_name)

        if latest_versions:
            model_version = latest_versions[0]
            registry_info = {
                "registry_name": registered_model_name,
                "version": model_version.version,
                "stage": model_version.current_stage,
                "source": model_version.source
            }
    except Exception as e:
        logging.warning(f"Could not fetch MLflow registry info: {e}")

    return {
        "model_name": model_info['model_name'],
        "model_type": model_info['model_type'],
        "training_date": model_info['training_date'],
        "feature_count": len(model_info['feature_columns']),
        "features": model_info['feature_columns'],
        "mlflow_registry": registry_info
    }

@app.post("/admin/reload-model")
async def reload_model():
    """Reload model from MLflow Registry (useful after model promotion)."""
    try:
        load_model()
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)