"""
Entrenamiento YOLO11m — servidor remoto, GPU única, 640px.

Uso:
    python3 compvision/scripts/train_yolo11m_640.py
    GPU_DEVICE=0 python3 compvision/scripts/train_yolo11m_640.py
"""
import os
from pathlib import Path

import yaml
from ultralytics import YOLO

SCRIPTS_DIR  = Path(__file__).parent
HYP_FILE     = SCRIPTS_DIR / "hyp_yolo11m_960.yaml"
DATASET_YAML = SCRIPTS_DIR.parent.parent / "dataset_yolo" / "dataset.yaml"
DEVICE       = int(os.environ.get("GPU_DEVICE", 0))

MLFLOW_URI  = os.environ.get("MLFLOW_TRACKING_URI",    "http://172.17.0.1:5050")
EXPERIMENT  = os.environ.get("MLFLOW_EXPERIMENT_NAME", "vision_sdc")


def setup_mlflow():
    os.environ["MLFLOW_TRACKING_URI"]    = MLFLOW_URI
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT
    print(f"[MLflow] tracking → {MLFLOW_URI}  |  experiment: {EXPERIMENT}")


def train():
    setup_mlflow()

    with open(HYP_FILE, encoding="utf-8") as f:
        hyp = yaml.safe_load(f)

    model_weights = hyp.pop("model", "yolo11m.pt")

    print(f"YOLO11m — {hyp['imgsz']}px · batch={hyp['batch']} · device={DEVICE}")
    print(f"Hiperparámetros : {HYP_FILE.name}")
    print(f"Experimento     : {hyp['name']}")
    print(f"Dataset         : {DATASET_YAML}")

    model = YOLO(model_weights)

    model.train(
        data=str(DATASET_YAML),
        device=DEVICE,
        **{k: v for k, v in hyp.items() if k not in ("project", "name")},
        project=hyp["project"],
        name=hyp["name"],
        verbose=True,
    )

    print("\n" + "=" * 40)
    print("YOLO11m COMPLETADO.")
    print(f"Mejor modelo: runs/detect/{hyp['project']}/{hyp['name']}/weights/best.pt")
    print("=" * 40)


if __name__ == "__main__":
    train()
