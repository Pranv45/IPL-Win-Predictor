#!/bin/bash
set -e

# Fix ownership of mounted volumes if running as root
if [ "$(id -u)" = "0" ]; then
  echo "Running as root, fixing permissions..."
  chown -R airflow:root /opt/airflow/logs /opt/airflow/data /opt/airflow/models /opt/airflow/metrics 2>/dev/null || true
  echo "Switching to airflow user..."
  exec gosu airflow "$0" "$@"
fi

echo "Waiting for database to be ready..."
until airflow db check; do
  echo "Database not ready, waiting..."
  sleep 2
done

echo "Database is ready!"

# Only create admin user from webserver (not scheduler)
if [ "${AIRFLOW_ROLE}" = "webserver" ]; then
  echo "Creating admin user if it doesn't exist..."
  airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin \
    2>&1 | grep -v "already exists" || true

  echo "Starting webserver..."
  exec airflow webserver
elif [ "${AIRFLOW_ROLE}" = "scheduler" ]; then
  echo "Starting scheduler..."
  exec airflow scheduler
else
  echo "Error: AIRFLOW_ROLE environment variable must be set to 'webserver' or 'scheduler'"
  exit 1
fi

