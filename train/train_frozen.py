import os
import argparse
from ultralytics import YOLO
from mlflow_setup import setup as mlflow_setup

def train_frozen(base_model='yolov8s.pt'):
    mlflow_setup()
    model = YOLO(base_model)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_yaml = os.path.abspath(os.path.join(script_dir, '..', 'dataset_yolo', 'dataset.yaml'))

    arch = base_model.replace('.pt', '')
    print(f"Iniciando Transfer Learning (backbone congelado) con {arch}: {dataset_yaml}")
    print("Capas 0-9 congeladas (backbone). Solo se entrena el neck y la cabeza de detección.")

    results = model.train(
        data=dataset_yaml,
        epochs=30,
        imgsz=640,
        batch=32,
        workers=4,
        device=0,
        amp=True,
        freeze=10,           # Congela las 10 capas del backbone (0-9)
        project='vision_proyect',
        name=f'frozen_{arch}',
        optimizer='AdamW',
        lr0=0.01,            # lr más alto: solo se actualiza la cabeza
        cos_lr=True,
        patience=10,
        verbose=True
    )

    print("\n" + "="*30)
    print("Transfer Learning (frozen) COMPLETADO.")
    print(f"Mejor modelo guardado en: {results.save_dir}")
    print("="*30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8s.pt', help='Modelo base (yolov8s.pt o yolo11s.pt)')
    args = parser.parse_args()
    train_frozen(args.model)
