#!/bin/bash
set -e

echo "Waiting for database to be ready..."
until airflow db check; do
  echo "Database not ready, waiting..."
  sleep 2
done

echo "Database is ready!"

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

