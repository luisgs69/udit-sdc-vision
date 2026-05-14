import os
from pathlib import Path
import yaml
from ultralytics import YOLO
from mlflow_setup import setup as mlflow_setup

HYP_FILE = Path(__file__).parent / "hyp_yolo11s_960.yaml"
DEVICE    = int(os.environ.get("GPU_DEVICE", 2))


def train():
    mlflow_setup()

    with open(HYP_FILE, encoding="utf-8") as f:
        hyp = yaml.safe_load(f)

    dataset_yaml = Path(__file__).parent.parent / "dataset_yolo" / "dataset.yaml"

    print(f"YOLO11s finetuning — {hyp['imgsz']}px · batch={hyp['batch']} · device={DEVICE}")
    print(f"Hiperparámetros: {HYP_FILE}")
    print(f"Dataset:         {dataset_yaml}")

    model = YOLO("yolo11s.pt")

    model.train(
        data=str(dataset_yaml),
        device=DEVICE,
        **{k: v for k, v in hyp.items() if k not in ("project", "name")},
        project=hyp["project"],
        name=hyp["name"],
        mlflow=True,
        verbose=True,
    )

    save_dir = Path("runs/detect") / hyp["project"] / hyp["name"]
    print("\n" + "=" * 40)
    print("YOLO11s COMPLETADO.")
    print(f"Mejor modelo: {save_dir}/weights/best.pt")
    print("=" * 40)


if __name__ == "__main__":
    train()
