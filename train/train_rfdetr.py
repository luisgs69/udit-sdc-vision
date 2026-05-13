import os
from pathlib import Path
from rfdetr import RFDETRBase
from mlflow_setup import setup as mlflow_setup, MLFLOW_URI, EXPERIMENT

def train_rfdetr():
    mlflow_setup()

    script_dir  = Path(__file__).parent
    dataset_dir = script_dir.parent / "dataset_coco"
    output_dir  = "/ultralytics/runs/detect/vision_proyect/finetuning_rfdetr_base"

    print(f"Iniciando Fine-Tuning RF-DETR-base (DINOv2 backbone): {dataset_dir}")

    model = RFDETRBase()

    model.train(
        dataset_dir=str(dataset_dir),
        epochs=30,
        batch_size=8,           # A2 16GB: base cabe con batch=8
        grad_accum_steps=4,     # gradiente efectivo = 32 imágenes
        lr=1e-4,
        lr_encoder=1e-5,        # backbone DINOv2 necesita lr más bajo
        output_dir=output_dir,
        mlflow=True,
        project=EXPERIMENT,
        run="finetuning_rfdetr_base",
    )

    print("\n" + "="*30)
    print("Fine-Tuning RF-DETR-base COMPLETADO.")
    print(f"Mejor modelo guardado en: {output_dir}/best.pth")
    print("="*30)

if __name__ == "__main__":
    train_rfdetr()
