#!/bin/bash
# Arranca el servidor MLflow en el host en el puerto 5050.
# Accesible desde los containers via http://172.17.0.1:5050
# Accesible desde el navegador en http://localhost:5050

MLRUNS_DIR="$(dirname "$0")/mlruns"
mkdir -p "$MLRUNS_DIR"

echo "Iniciando MLflow en http://0.0.0.0:5050 (datos en $MLRUNS_DIR)"
mlflow server \
  --backend-store-uri "sqlite:///$MLRUNS_DIR/mlflow.db" \
  --default-artifact-root "$MLRUNS_DIR/artifacts" \
  --host 0.0.0.0 \
  --port 5050
