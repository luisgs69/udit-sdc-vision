import os

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://172.17.0.1:5050")
EXPERIMENT  = os.environ.get("MLFLOW_EXPERIMENT_NAME", "vision_sdc")


def setup():
    """Configura las variables de entorno que ultralytics y rfdetr leen automáticamente."""
    os.environ["MLFLOW_TRACKING_URI"]     = MLFLOW_URI
    os.environ["MLFLOW_EXPERIMENT_NAME"]  = EXPERIMENT
    print(f"[MLflow] tracking → {MLFLOW_URI}  |  experiment: {EXPERIMENT}")
