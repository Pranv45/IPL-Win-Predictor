#!/usr/bin/env python3
"""
IPL Win Predictor - Startup Script
This script starts the API server for the IPL Win Predictor application.
"""

import subprocess
import sys
import os
import time
import requests
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import pandas
        import numpy
        import sklearn
        import xgboost
        import fastapi
        import uvicorn
        import mlflow
        print("✓ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_model_files():
    """Check if model files exist."""
    required_files = [
        "models/ipl_win_predictor.pkl",
        "models/scaler.pkl",
        "models/model_info.json",
        "data/features/label_encoders.pkl"
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("✗ Missing model files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease run the training pipeline first:")
        print("  python scripts/ingest_and_process.py")
        print("  python scripts/feature_engineering.py")
        print("  python scripts/train_model.py")
        return False

    print("✓ All model files are present")
    return True

def start_api_server():
    """Start the API server."""
    print("\n🚀 Starting IPL Win Predictor API Server...")
    print("API will be available at: http://localhost:8080")
    print("API Documentation: http://localhost:8080/docs")
    print("Health Check: http://localhost:8080/health")
    print("\nPress Ctrl+C to stop the server")

    try:
        # Start the API server
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8080",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 API server stopped. Goodbye!")

def main():
    """Main function."""
    print("🏏 IPL Win Predictor - Startup Script")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check model files
    if not check_model_files():
        sys.exit(1)

    # Start the API server
    start_api_server()

if __name__ == "__main__":
    main()
