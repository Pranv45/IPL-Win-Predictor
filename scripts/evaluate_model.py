import pandas as pd
import numpy as np
import yaml
import logging
import os
import json
import pickle
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    """Load model configuration."""
    with open('configs/model_config.yaml', 'r') as file:
        return yaml.safe_load(file)

def load_model_and_data():
    """Load trained model and test data."""
    config = load_config()

    # Load model
    model_path = f"models/{config['model']['name']}.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load scaler
    scaler_path = "models/scaler.pkl"
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Load model info
    with open('models/model_info.json', 'r') as f:
        model_info = json.load(f)

    # Load features
    features_path = "data/features/engineered_features.parquet"
    df = pd.read_parquet(features_path)

    return model, scaler, model_info, df

def prepare_test_data(df, model_info, scaler):
    """Prepare test data for evaluation."""
    logging.info("Preparing test data for evaluation...")

    # Select features
    feature_columns = model_info['feature_columns']
    X = df[feature_columns].fillna(0)
    y = df['winner']

    # Scale features
    X_scaled = scaler.transform(X)

    return X_scaled, y

def comprehensive_evaluation(model, X_test, y_test, config):
    """Perform comprehensive model evaluation."""
    logging.info("Performing comprehensive model evaluation...")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        roc_auc = None

    # Cross-validation score
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')

    # Calculate additional metrics
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    evaluation_results = {
        'accuracy': accuracy,
        'precision': classification_rep['weighted avg']['precision'],
        'recall': classification_rep['weighted avg']['recall'],
        'f1_score': classification_rep['weighted avg']['f1-score'],
        'roc_auc': roc_auc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix.tolist(),
        'cv_scores': cv_scores.tolist(),
        'timestamp': datetime.now().isoformat()
    }

    return evaluation_results

def create_evaluation_plots(y_test, y_pred, y_pred_proba, output_dir, model):
    """Create evaluation plots."""
    logging.info("Creating evaluation plots...")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. ROC Curve
    if y_pred_proba is not None:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        feature_importance = pd.DataFrame({
            'feature': range(len(model.feature_importances_)),
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)

        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), [f'Feature {i}' for i in feature_importance['feature']])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_html_report(evaluation_results, output_dir):
    """Generate HTML evaluation report."""
    logging.info("Generating HTML evaluation report...")

    # Fix ROC AUC formatting
    roc_auc_val = evaluation_results.get('roc_auc')
    if roc_auc_val is not None:
        roc_auc_str = f"{roc_auc_val:.4f}"
    else:
        roc_auc_str = 'N/A'

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>IPL Win Predictor - Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ color: #7f8c8d; }}
            .section {{ margin: 30px 0; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>IPL Win Predictor - Model Evaluation Report</h1>
        <p><strong>Generated:</strong> {evaluation_results['timestamp']}</p>

        <div class="section">
            <h2>Model Performance Metrics</h2>
            <div class="metric">
                <div class="metric-value">{evaluation_results['accuracy']:.4f}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric">
                <div class="metric-value">{evaluation_results['precision']:.4f}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric">
                <div class="metric-value">{evaluation_results['recall']:.4f}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric">
                <div class="metric-value">{evaluation_results['f1_score']:.4f}</div>
                <div class="metric-label">F1 Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{roc_auc_str}</div>
                <div class="metric-label">ROC AUC</div>
            </div>
        </div>

        <div class="section">
            <h2>Cross-Validation Results</h2>
            <div class="metric">
                <div class="metric-value">{evaluation_results['cv_mean']:.4f} ± {evaluation_results['cv_std']:.4f}</div>
                <div class="metric-label">CV Accuracy (Mean ± Std)</div>
            </div>
        </div>

        <div class="section">
            <h2>Visualizations</h2>
            <img src="confusion_matrix.png" alt="Confusion Matrix">
            <img src="roc_curve.png" alt="ROC Curve">
            <img src="feature_importance.png" alt="Feature Importance">
        </div>

        <div class="section">
            <h2>Detailed Classification Report</h2>
            <pre>{json.dumps(evaluation_results['classification_report'], indent=2)}</pre>
        </div>
    </body>
    </html>
    """

    with open(f'{output_dir}/model_evaluation.html', 'w') as f:
        f.write(html_content)


def log_evaluation_to_mlflow(evaluation_results: dict, config: dict) -> None:
    """
    Log evaluation metrics to MLflow.

    Uses the same tracking URI / experiment as training, and creates a
    dedicated run named 'model_evaluation'. Any MLflow-related errors are
    logged but do NOT fail the evaluation pipeline.
    """
    try:
        mlflow_config = config.get("mlflow", {})
        # Check environment variable first (for Docker), then config
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', mlflow_config.get("tracking_uri"))
        experiment_name = mlflow_config.get("experiment_name", "ipl_win_prediction")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logging.info(f"MLflow tracking URI: {tracking_uri}")
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="model_evaluation"):
            # Core metrics
            mlflow.log_metric("accuracy", evaluation_results.get("accuracy", 0.0))
            mlflow.log_metric("precision", evaluation_results.get("precision", 0.0))
            mlflow.log_metric("recall", evaluation_results.get("recall", 0.0))
            mlflow.log_metric("f1_score", evaluation_results.get("f1_score", 0.0))

            # Optional metrics
            if evaluation_results.get("roc_auc") is not None:
                mlflow.log_metric("roc_auc", evaluation_results["roc_auc"])
            mlflow.log_metric("specificity", evaluation_results.get("specificity", 0.0))
            mlflow.log_metric("sensitivity", evaluation_results.get("sensitivity", 0.0))
            mlflow.log_metric("cv_mean", evaluation_results.get("cv_mean", 0.0))
            mlflow.log_metric("cv_std", evaluation_results.get("cv_std", 0.0))

            # Log full classification report & confusion matrix as artifacts
            mlflow.log_dict(
                evaluation_results.get("classification_report", {}),
                "classification_report.json",
            )
            mlflow.log_dict(
                {"confusion_matrix": evaluation_results.get("confusion_matrix", [])},
                "confusion_matrix.json",
            )

        logging.info("Evaluation metrics logged to MLflow successfully.")
    except Exception as e:
        # Do not break the pipeline if MLflow is misconfigured or unavailable
        logging.warning(f"Failed to log evaluation to MLflow: {e}")


def main():
    """Main evaluation pipeline."""
    try:
        # Load configuration
        config = load_config()

        # Load model and data
        model, scaler, model_info, df = load_model_and_data()

        # Prepare test data
        X_test, y_test = prepare_test_data(df, model_info, scaler)

        # Perform evaluation
        evaluation_results = comprehensive_evaluation(model, X_test, y_test, config)

        # Create plots
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        create_evaluation_plots(y_test, y_pred, y_pred_proba, "reports", model)

        # Generate HTML report
        generate_html_report(evaluation_results, "reports")

        # Save evaluation results
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/evaluation_results.json", "w") as f:
            json.dump(evaluation_results, f, indent=2)

        # Log to MLflow (best-effort)
        log_evaluation_to_mlflow(evaluation_results, config)

        logging.info("Model evaluation completed successfully!")
        logging.info(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        logging.info(f"F1 Score: {evaluation_results['f1_score']:.4f}")

    except Exception as e:
        logging.error(f"Error in evaluation pipeline: {e}")
        raise


if __name__ == "__main__":
    main()