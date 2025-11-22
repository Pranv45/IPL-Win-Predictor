#!/usr/bin/env python3
"""
Pipeline Health Check Script
Validates that all components are properly configured before running the DAG.
"""
import os
import sys
import yaml
import json
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists."""
    if os.path.isdir(dirpath):
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {dirpath}")
        return False

def validate_config_file(config_path):
    """Validate the model config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check required sections
        required_sections = ['model', 'features', 'training', 'hyperparameters', 'mlflow']
        missing = [s for s in required_sections if s not in config]

        if missing:
            print(f"❌ Config missing sections: {missing}")
            return False

        # Check required feature keys
        if 'numerical_features' not in config['features']:
            print("❌ Config missing 'numerical_features'")
            return False
        if 'categorical_features' not in config['features']:
            print("❌ Config missing 'categorical_features'")
            return False
        if 'target_column' not in config['features']:
            print("❌ Config missing 'target_column'")
            return False

        print(f"✅ Config file is valid: {config_path}")
        return True
    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        return False

def validate_dvc_config():
    """Validate DVC configuration."""
    try:
        with open('dvc.yaml', 'r') as f:
            dvc_config = yaml.safe_load(f)

        # Check required stages
        required_stages = ['data_processing', 'feature_engineering', 'model_training', 'model_evaluation']
        stages = dvc_config.get('stages', {})

        missing_stages = [s for s in required_stages if s not in stages]
        if missing_stages:
            print(f"❌ DVC missing stages: {missing_stages}")
            return False

        # Validate each stage has required fields
        for stage_name in required_stages:
            stage = stages[stage_name]
            if 'cmd' not in stage:
                print(f"❌ DVC stage '{stage_name}' missing 'cmd'")
                return False
            if 'deps' not in stage:
                print(f"⚠️  DVC stage '{stage_name}' missing 'deps' (optional but recommended)")

        print("✅ DVC configuration is valid")
        return True
    except Exception as e:
        print(f"❌ DVC validation failed: {e}")
        return False

def main():
    """Run all health checks."""
    print("=" * 60)
    print("🔍 IPL Win Predictor - Pipeline Health Check")
    print("=" * 60)
    print()

    all_passed = True

    # Check raw data files
    print("📁 Checking Raw Data Files...")
    all_passed &= check_file_exists('data/raw/matches.csv', 'Matches CSV')
    all_passed &= check_file_exists('data/raw/deliveries.csv', 'Deliveries CSV')
    print()

    # Check directories
    print("📁 Checking Required Directories...")
    required_dirs = [
        ('data/processed', 'Processed Data Directory'),
        ('data/features', 'Features Directory'),
        ('models', 'Models Directory'),
        ('metrics', 'Metrics Directory'),
        ('configs', 'Configs Directory'),
        ('scripts', 'Scripts Directory'),
        ('dags', 'DAGs Directory'),
    ]

    for dirpath, description in required_dirs:
        if not os.path.exists(dirpath):
            print(f"⚠️  Creating missing directory: {dirpath}")
            os.makedirs(dirpath, exist_ok=True)
        all_passed &= check_directory_exists(dirpath, description)
    print()

    # Check script files
    print("🐍 Checking Python Scripts...")
    scripts = [
        ('scripts/ingest_and_process.py', 'Data Processing Script'),
        ('scripts/feature_engineering.py', 'Feature Engineering Script'),
        ('scripts/train_model.py', 'Training Script'),
        ('scripts/evaluate_model.py', 'Evaluation Script'),
        ('dags/ml_pipeline_dag.py', 'Airflow DAG'),
    ]

    for script_path, description in scripts:
        all_passed &= check_file_exists(script_path, description)
    print()

    # Check configuration files
    print("⚙️  Checking Configuration Files...")
    all_passed &= check_file_exists('configs/model_config.yaml', 'Model Config')
    all_passed &= check_file_exists('dvc.yaml', 'DVC Config')
    all_passed &= check_file_exists('docker-compose.yml', 'Docker Compose')
    print()

    # Validate configurations
    print("🔬 Validating Configuration Files...")
    all_passed &= validate_config_file('configs/model_config.yaml')
    all_passed &= validate_dvc_config()
    print()

    # Check Docker files
    print("🐳 Checking Docker Files...")
    all_passed &= check_file_exists('Dockerfile.airflow', 'Airflow Dockerfile')
    all_passed &= check_file_exists('Dockerfile.api', 'API Dockerfile')
    all_passed &= check_file_exists('airflow-entrypoint.sh', 'Airflow Entrypoint')
    print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("✅ All health checks PASSED! Pipeline is ready to run.")
        print("=" * 60)
        print()
        print("🚀 To run the pipeline:")
        print("   1. Start services: docker compose up -d")
        print("   2. Open Airflow UI: http://localhost:8081")
        print("   3. Login with: admin / admin")
        print("   4. Enable and trigger the 'ipl_ml_pipeline' DAG")
        return 0
    else:
        print("❌ Some health checks FAILED! Please fix the issues above.")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())

