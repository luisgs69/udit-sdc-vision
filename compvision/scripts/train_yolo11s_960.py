"""
Uso:
    python3 compvision/scripts/train_yolo11s_960.py                          # base (sin aug)
    python3 compvision/scripts/train_yolo11s_960.py --hyp hyp_yolo11s_960_aug.yaml  # con aug
    GPU_DEVICE=1 python3 compvision/scripts/train_yolo11s_960.py
"""
import argparse
import os
from pathlib import Path

import yaml
from ultralytics import YOLO

SCRIPTS_DIR  = Path(__file__).parent
DATASET_YAML = SCRIPTS_DIR.parent.parent / "dataset_yolo" / "dataset.yaml"
DEVICE       = int(os.environ.get("GPU_DEVICE", 2))

MLFLOW_URI  = os.environ.get("MLFLOW_TRACKING_URI",    "http://172.17.0.1:5050")
EXPERIMENT  = os.environ.get("MLFLOW_EXPERIMENT_NAME", "vision_sdc")


def setup_mlflow():
    os.environ["MLFLOW_TRACKING_URI"]    = MLFLOW_URI
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT
    print(f"[MLflow] tracking → {MLFLOW_URI}  |  experiment: {EXPERIMENT}")


def train(hyp_file: Path):
    setup_mlflow()

    with open(hyp_file, encoding="utf-8") as f:
        hyp = yaml.safe_load(f)

    print(f"YOLO11s — {hyp['imgsz']}px · batch={hyp['batch']} · device={DEVICE}")
    print(f"Hiperparámetros : {hyp_file.name}")
    print(f"Experimento     : {hyp['name']}")
    print(f"Dataset         : {DATASET_YAML}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyp",
        default="hyp_yolo11s_960_base.yaml",
        help="Nombre del YAML de hiperparámetros (buscado en compvision/scripts/)",
    )
    args = parser.parse_args()

    hyp_path = SCRIPTS_DIR / args.hyp
    if not hyp_path.exists():
        raise FileNotFoundError(f"No se encuentra {hyp_path}")

    train(hyp_path)
