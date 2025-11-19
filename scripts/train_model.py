import pandas as pd
import numpy as np
import yaml
import logging
import os
import json
from datetime import datetime
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    """Load model configuration."""
    with open('configs/model_config.yaml', 'r') as file:
        return yaml.safe_load(file)

def load_features():
    """Load engineered features."""
    features_path = "data/features/engineered_features.parquet"
    df = pd.read_parquet(features_path)
    logging.info(f"Loaded features with shape: {df.shape}")
    return df

def prepare_data(df, config):
    """Prepare data for training."""
    logging.info("Preparing data for training...")

    # Select features for training
    feature_columns = []

    # Add numerical features
    for feature in config['features']['numerical_features']:
        if feature in df.columns:
            feature_columns.append(feature)

    # Add encoded categorical features
    for feature in config['features']['categorical_features']:
        encoded_col = f'{feature}_encoded'
        if encoded_col in df.columns:
            feature_columns.append(encoded_col)

    # Add interaction features
    interaction_features = ['win_percentage_diff', 'recent_form_diff', 'h2h_advantage']
    for feature in interaction_features:
        if feature in df.columns:
            feature_columns.append(feature)

    logging.info(f"Selected {len(feature_columns)} features for training")

    # Prepare X and y
    X = df[feature_columns].fillna(0)
    y = df[config['features']['target_column']]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns

def train_model(X_train, y_train, X_test, y_test, scaler, feature_columns, config):
    """Train the model with MLflow tracking and register to Model Registry."""
    logging.info("Starting model training with MLflow...")

    # Set up MLflow - check environment variable first (for Docker), then config
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', config['mlflow']['tracking_uri'])
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logging.info(f"MLflow tracking URI: {mlflow_tracking_uri}")
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(config['hyperparameters'])

        # Train model
        model = xgb.XGBClassifier(**config['hyperparameters'])
        model.fit(X_train, y_train)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log feature importance
        feature_importance = dict(zip(range(len(model.feature_importances_)), model.feature_importances_))
        mlflow.log_dict(feature_importance, "feature_importance.json")

        # Evaluate model within the same run context
        metrics = evaluate_model(model, X_test, y_test, config)

        # Save scaler and model info as artifacts (needed for API to load from registry)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save scaler
            scaler_path = os.path.join(tmpdir, "scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(scaler_path, "artifacts")

            # Get DVC data version info (if available) - links model to versioned data
            dvc_info = {}
            try:
                # Get DVC lock file info to track data versions
                dvc_lock_path = 'dvc.lock'
                if os.path.exists(dvc_lock_path):
                    with open(dvc_lock_path, 'r') as f:
                        dvc_lock = yaml.safe_load(f)
                        # Extract stage info from DVC lock file
                        stages = dvc_lock.get('stages', {})

                        # Get data processing output hash
                        data_proc_outs = stages.get('data_processing', {}).get('outs', [])
                        data_proc_hash = data_proc_outs[0].get('hash', 'unknown') if data_proc_outs else 'unknown'

                        # Get feature engineering output hash
                        feat_eng_outs = stages.get('feature_engineering', {}).get('outs', [])
                        feat_eng_hash = feat_eng_outs[0].get('hash', 'unknown') if feat_eng_outs else 'unknown'

                        # Get training data dependency hash (from feature engineering output)
                        model_train_deps = stages.get('model_training', {}).get('deps', [])
                        train_data_hash = model_train_deps[0].get('hash', 'unknown') if model_train_deps else feat_eng_hash

                        dvc_info = {
                            'data_processing_hash': data_proc_hash,
                            'feature_engineering_hash': feat_eng_hash,
                            'training_data_hash': train_data_hash
                        }
                        logging.info(f"DVC data version hashes extracted: {dvc_info}")
            except Exception as e:
                logging.warning(f"Could not extract DVC version info: {e}")

            # Save model info with DVC data version tracking
            model_info = {
                'feature_columns': feature_columns,
                'model_name': config['model']['name'],
                'model_type': config['model']['type'],
                'training_date': datetime.now().isoformat(),
                'dvc_data_versions': dvc_info  # Link to DVC versioned data
            }
            model_info_path = os.path.join(tmpdir, "model_info.json")
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            mlflow.log_artifact(model_info_path, "artifacts")

        logging.info("Scaler and model info saved to MLflow artifacts")

        # Register model to MLflow Model Registry
        registered_model_name = config['mlflow'].get('registered_model_name', 'IPLWinPredictor')
        model_uri = f"runs:/{run.info.run_id}/model"

        try:
            client = MlflowClient()
            # Register the model (creates new version)
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name
            )
            logging.info(f"Model registered to MLflow Registry: {registered_model_name}, Version: {model_version.version}")

            # Add DVC data version info as model version description/tags
            if dvc_info:
                try:
                    # Add data version info as tags
                    client.set_model_version_tag(
                        name=registered_model_name,
                        version=model_version.version,
                        key="dvc_data_processing_hash",
                        value=dvc_info.get('data_processing_hash', 'unknown')
                    )
                    client.set_model_version_tag(
                        name=registered_model_name,
                        version=model_version.version,
                        key="dvc_feature_hash",
                        value=dvc_info.get('feature_engineering_hash', 'unknown')
                    )
                    client.set_model_version_tag(
                        name=registered_model_name,
                        version=model_version.version,
                        key="dvc_training_data_hash",
                        value=dvc_info.get('training_data_hash', 'unknown')
                    )
                    logging.info(f"DVC data version info linked to model version {model_version.version}")
                except Exception as e:
                    logging.warning(f"Could not add DVC tags to model version: {e}")

            # Transition to Staging by default (can be promoted to Production after validation)
            client.transition_model_version_stage(
                name=registered_model_name,
                version=model_version.version,
                stage="Staging"
            )
            logging.info(f"Model version {model_version.version} transitioned to Staging stage")

        except Exception as e:
            logging.warning(f"Failed to register model to registry: {e}. Model logged but not registered.")
            model_version = None

        logging.info("Model training completed successfully")
        return model, run.info.run_id, metrics

def evaluate_model(model, X_test, y_test, config):
    """Evaluate the model and log metrics."""
    logging.info("Evaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", classification_rep['weighted avg']['precision'])
    mlflow.log_metric("recall", classification_rep['weighted avg']['recall'])
    mlflow.log_metric("f1_score", classification_rep['weighted avg']['f1-score'])

    # Save metrics to file
    metrics = {
        'accuracy': accuracy,
        'precision': classification_rep['weighted avg']['precision'],
        'recall': classification_rep['weighted avg']['recall'],
        'f1_score': classification_rep['weighted avg']['f1-score'],
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix.tolist(),
        'timestamp': datetime.now().isoformat()
    }

    os.makedirs('metrics', exist_ok=True)
    with open('metrics/model_performance.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
    return metrics

def save_model(model, scaler, feature_columns, config):
    """Save the trained model and artifacts."""
    logging.info("Saving model and artifacts...")

    os.makedirs('models', exist_ok=True)

    # Save model
    model_path = f"models/{config['model']['name']}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save scaler
    scaler_path = "models/scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save feature columns
    feature_info = {
        'feature_columns': feature_columns,
        'model_name': config['model']['name'],
        'model_type': config['model']['type'],
        'training_date': datetime.now().isoformat()
    }

    with open('models/model_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)

    logging.info(f"Model saved to {model_path}")

def main():
    """Main training pipeline."""
    try:
        # Load configuration
        config = load_config()

        # Load features
        df = load_features()

        # Prepare data
        X_train, X_test, y_train, y_test, scaler, feature_columns = prepare_data(df, config)

        # Train model (returns model, run_id, and metrics)
        model, run_id, metrics = train_model(X_train, y_train, X_test, y_test, scaler, feature_columns, config)

        # Save model (still needed for backward compatibility)
        # Note: Scaler and model_info are already saved to MLflow artifacts in train_model()
        save_model(model, scaler, feature_columns, config)

        logging.info("Training pipeline completed successfully!")

    except Exception as e:
        logging.error(f"Error in training pipeline: {e}")
        raise

if __name__ == "__main__":
    main()