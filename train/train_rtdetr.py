import os
from ultralytics import YOLO
from mlflow_setup import setup as mlflow_setup

def train_rtdetr():
    mlflow_setup()
    # rtdetr-l.pt se descarga automáticamente si no está presente
    model = YOLO('rtdetr-l.pt')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_yaml = os.path.abspath(os.path.join(script_dir, '..', 'dataset_yolo', 'dataset.yaml'))

    print(f"Iniciando Fine-Tuning RT-DETR-l (Transformer decoder): {dataset_yaml}")

    results = model.train(
        data=dataset_yaml,
        epochs=30,
        imgsz=640,
        batch=8,             # A2 16GB: batch=16 OOM en deformable attention, batch=8 seguro
        workers=4,
        device=0,
        amp=True,
        project='vision_proyect',
        name='finetuning_rtdetr_l',
        optimizer='AdamW',
        lr0=0.0001,          # lr más bajo: transformers son sensibles a lr altos
        cos_lr=True,
        patience=10,
        verbose=True
    )

    print("\n" + "="*30)
    print("Fine-Tuning RT-DETR-l COMPLETADO.")
    print(f"Mejor modelo guardado en: {results.save_dir}")
    print("="*30)

if __name__ == '__main__':
    train_rtdetr()
