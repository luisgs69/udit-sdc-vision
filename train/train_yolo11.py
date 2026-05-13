import os
from ultralytics import YOLO
from mlflow_setup import setup as mlflow_setup

def train_yolo11():
    mlflow_setup()
    # yolo11s.pt se descarga automáticamente si no está en el directorio
    model = YOLO('yolo11s.pt')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_yaml = os.path.abspath(os.path.join(script_dir, '..', 'dataset_yolo', 'dataset.yaml'))

    print(f"Iniciando Fine-Tuning YOLO11s (Dual Label Assignment): {dataset_yaml}")

    results = model.train(
        data=dataset_yaml,
        epochs=30,
        imgsz=640,
        batch=32,
        workers=4,
        device=0,
        amp=True,
        project='vision_proyect',
        name='finetuning_yolo11s',
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        patience=10,
        verbose=True
    )

    print("\n" + "="*30)
    print("Fine-Tuning YOLO11s COMPLETADO.")
    print(f"Mejor modelo guardado en: {results.save_dir}")
    print("="*30)

if __name__ == '__main__':
    train_yolo11()
