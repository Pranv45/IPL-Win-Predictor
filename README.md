# IPL Win Predictor MLOps Project

This project implements an end-to-end MLOps pipeline to train, deploy, and monitor an IPL win prediction model, following best practices for software engineering and production-grade machine learning.

## 1. Data Ingestion and Processing

This initial step uses Apache Spark to ingest the raw IPL data, clean it, perform feature engineering, and save the result as a Parquet file. The entire Spark environment is containerized using Docker to ensure consistency and reproducibility.

### Prerequisites

- Docker Desktop must be installed and running on your local machine.

### How to Run

1.  **Build the Spark Environment Docker Image:**
    This command builds a Docker image tagged `spark-env:latest` using the instructions specified in `Dockerfile.spark`. This image contains all necessary dependencies, including Python, Java, and Apache Spark.

    ```bash
    docker build -t spark-env:latest -f Dockerfile.spark .
    ```

2.  **Run the Spark Processing Script:**
    This command executes the `ingest_and_process.py` script inside a new container. It mounts the current project directory into the container's `/app` folder, which allows the script to read the raw data from `data/raw` and save the processed output to `data/processed`.

    ```bash
    docker run --rm -v "$(pwd):/app" spark-env:latest spark-submit scripts/ingest_and_process.py
    ```

### Expected Outcome

After the script runs successfully, a new directory will be created at `data/processed/processed_ipl_data.parquet`, containing the final, feature-engineered dataset ready for model training.

---