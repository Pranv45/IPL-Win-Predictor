from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ipl_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for IPL Win Predictor - Auto-retrains on data/config changes via DVC',
    schedule_interval='@daily',  # Runs daily, but DVC will skip stages if nothing changed
    catchup=False,
    tags=['mlops', 'ipl', 'prediction'],
)

def check_data_quality():
    """Check data quality before processing."""
    import pandas as pd
    import logging

    def to_py(obj):
        """Convert numpy/pandas objects to native Python types."""
        return obj.item() if hasattr(obj, "item") else obj

    try:
        # Check raw data files exist
        matches_path = '/opt/airflow/data/raw/matches.csv'
        deliveries_path = '/opt/airflow/data/raw/deliveries.csv'

        if not os.path.exists(matches_path):
            raise FileNotFoundError(f"Matches file not found: {matches_path}")
        if not os.path.exists(deliveries_path):
            raise FileNotFoundError(f"Deliveries file not found: {deliveries_path}")

        # Check raw data
        matches_df = pd.read_csv(matches_path)
        deliveries_df = pd.read_csv(deliveries_path)

        # Validate minimum data requirements
        if len(matches_df) == 0:
            raise ValueError("Matches dataset is empty")
        if len(deliveries_df) == 0:
            raise ValueError("Deliveries dataset is empty")

        # Basic quality checks
        quality_metrics = {
            'matches_count': len(matches_df),
            'deliveries_count': len(deliveries_df),
            'matches_missing_values': matches_df.isnull().sum().sum(),
            'deliveries_missing_values': deliveries_df.isnull().sum().sum(),
            'timestamp': datetime.now().isoformat()
        }
        quality_metrics = {k: to_py(v) for k, v in quality_metrics.items()}

        # Save quality metrics
        import json
        os.makedirs('/opt/airflow/metrics', exist_ok=True)
        with open('/opt/airflow/metrics/data_quality.json', 'w') as f:
            json.dump(quality_metrics, f, indent=2)

        logging.info(f"✅ Data quality check completed: {quality_metrics}")
        return quality_metrics

    except Exception as e:
        logging.error(f"❌ Data quality check failed: {e}")
        raise

def run_dvc_pipeline():
    """Run DVC pipeline stages."""
    import subprocess
    import logging

    # Run DVC pipeline
    try:
        # Data processing
        subprocess.run(['dvc', 'repro', 'data_processing'], check=True, cwd='/opt/airflow')
        logging.info("Data processing stage completed")

        # Feature engineering
        subprocess.run(['dvc', 'repro', 'feature_engineering'], check=True, cwd='/opt/airflow')
        logging.info("Feature engineering stage completed")

        # Model training
        subprocess.run(['dvc', 'repro', 'model_training'], check=True, cwd='/opt/airflow')
        logging.info("Model training stage completed")

        # Model evaluation
        subprocess.run(['dvc', 'repro', 'model_evaluation'], check=True, cwd='/opt/airflow')
        logging.info("Model evaluation stage completed")

    except subprocess.CalledProcessError as e:
        logging.error(f"DVC pipeline failed: {e}")
        raise

def deploy_model():
    """Deploy the trained model."""
    import shutil
    import logging

    try:
        # Define paths
        model_source = '/opt/airflow/models/ipl_win_predictor.pkl'
        scaler_source = '/opt/airflow/models/scaler.pkl'
        model_info_source = '/opt/airflow/models/model_info.json'

        deployment_dir = '/opt/airflow/models/deployed'

        # Validate source files exist
        if not os.path.exists(model_source):
            raise FileNotFoundError(f"Model file not found: {model_source}")
        if not os.path.exists(scaler_source):
            raise FileNotFoundError(f"Scaler file not found: {scaler_source}")
        if not os.path.exists(model_info_source):
            raise FileNotFoundError(f"Model info file not found: {model_info_source}")

        # Create deployment directory
        os.makedirs(deployment_dir, exist_ok=True)

        # Copy model artifacts
        shutil.copy2(model_source, f'{deployment_dir}/ipl_win_predictor.pkl')
        shutil.copy2(scaler_source, f'{deployment_dir}/scaler.pkl')
        shutil.copy2(model_info_source, f'{deployment_dir}/model_info.json')

        # Verify deployment
        if not os.path.exists(f'{deployment_dir}/ipl_win_predictor.pkl'):
            raise RuntimeError("Model deployment verification failed")

        logging.info("✅ Model deployed successfully to {deployment_dir}")
        return "Model deployed"

    except Exception as e:
        logging.error(f"❌ Model deployment failed: {e}")
        raise

def send_notification():
    """Send notification about pipeline completion."""
    import logging
    logging.info("ML Pipeline completed successfully!")
    return "Pipeline completed"

# Define tasks
check_data_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag,
)

def run_dvc_stage(stage_name):
    """Run a specific DVC pipeline stage.

    DVC automatically detects if dependencies changed:
    - If dependencies changed → reruns the stage (automatic retraining)
    - If nothing changed → skips (uses cache)

    Dependencies tracked:
    - feature_engineering: scripts/feature_engineering.py, data/processed/*
    - model_training: scripts/train_model.py, configs/model_config.yaml, data/features/*

    This provides automatic retraining when:
    - Features are changed (feature_engineering.py modified)
    - Model config is changed (model_config.yaml modified)
    - Training script is changed (train_model.py modified)
    - Data changes (raw data or processed data changes)
    """
    import subprocess
    import logging

    logging.info(f"Running DVC stage: {stage_name}")
    logging.info(f"DVC will automatically detect if dependencies changed and rerun if needed")

    result = subprocess.run(
        ['dvc', 'repro', stage_name],
        cwd='/opt/airflow',
        capture_output=True,
        text=True
    )

    # Check if DVC actually ran or used cache
    output_text = result.stdout + result.stderr
    if 'Data and pipelines are up to date' in output_text or 'is up to date' in output_text:
        logging.info(f"✅ DVC stage {stage_name} - No changes detected, using cached results (no retraining needed)")
    elif 'Running' in output_text or 'Reproducing' in output_text:
        logging.info(f"🔄 DVC stage {stage_name} - CHANGES DETECTED! Stage is being executed (automatic retraining triggered)")
        # Log what changed if available
        if 'deps' in output_text.lower() or 'dependencies' in output_text.lower():
            logging.info(f"   → Dependencies changed, triggering {stage_name} stage")
    else:
        logging.info(f"🔄 DVC stage {stage_name} - Changes detected, stage was executed")

    if result.returncode != 0:
        logging.error(f"DVC stage {stage_name} failed: {result.stderr}")
        raise Exception(f"DVC stage {stage_name} failed: {result.stderr}")
    logging.info(f"DVC stage {stage_name} completed successfully")
    return f"DVC stage {stage_name} completed"

def run_data_processing():
    """Run data processing via DVC pipeline."""
    return run_dvc_stage('data_processing')

def run_feature_engineering():
    """Run feature engineering via DVC pipeline."""
    return run_dvc_stage('feature_engineering')

def run_model_training():
    """Run model training via DVC pipeline."""
    return run_dvc_stage('model_training')

def run_model_evaluation():
    """Run model evaluation via DVC pipeline."""
    return run_dvc_stage('model_evaluation')

data_processing_task = PythonOperator(
    task_id='data_processing',
    python_callable=run_data_processing,
    dag=dag,
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=run_feature_engineering,
    dag=dag,
)

model_training_task = PythonOperator(
    task_id='model_training',
    python_callable=run_model_training,
    dag=dag,
)

model_evaluation_task = PythonOperator(
    task_id='model_evaluation',
    python_callable=run_model_evaluation,
    dag=dag,
)

# DVC is now integrated - each task uses 'dvc repro' to execute the pipeline stage
# This provides both DVC versioning benefits and Airflow UI visibility

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

notification_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag,
)

# Define task dependencies
# Pipeline flow: data quality check -> processing -> feature engineering -> training -> evaluation -> deployment -> notification
check_data_task >> data_processing_task >> feature_engineering_task >> model_training_task >> model_evaluation_task >> deploy_model_task >> notification_task