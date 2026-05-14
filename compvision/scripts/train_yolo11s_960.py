import os
from pathlib import Path
import yaml
from ultralytics import YOLO

HYP_FILE     = Path(__file__).parent / "hyp_yolo11s_960.yaml"
DATASET_YAML = Path(__file__).parent.parent.parent / "dataset_yolo" / "dataset.yaml"
DEVICE       = int(os.environ.get("GPU_DEVICE", 2))

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

    print(f"YOLO11s finetuning — {hyp['imgsz']}px · batch={hyp['batch']} · device={DEVICE}")
    print(f"Hiperparámetros: {HYP_FILE}")
    print(f"Dataset:         {DATASET_YAML}")

    model = YOLO("yolo11s.pt")

    model.train(
        data=str(DATASET_YAML),
        device=DEVICE,
        **{k: v for k, v in hyp.items() if k not in ("project", "name")},
        project=hyp["project"],
        name=hyp["name"],
        mlflow=True,
        verbose=True,
    )

    print("\n" + "=" * 40)
    print("YOLO11s COMPLETADO.")
    print(f"Mejor modelo: runs/detect/{hyp['project']}/{hyp['name']}/weights/best.pt")
    print("=" * 40)


if __name__ == "__main__":
    train()
