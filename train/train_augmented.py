import os
import argparse
from ultralytics import YOLO
from mlflow_setup import setup as mlflow_setup

def train_augmented(base_model='yolov8s.pt'):
    mlflow_setup()
    model = YOLO(base_model)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_yaml = os.path.abspath(os.path.join(script_dir, '..', 'dataset_yolo', 'dataset.yaml'))

    arch = base_model.replace('.pt', '')
    print(f"Iniciando Fine-Tuning con Data Augmentation agresiva ({arch}): {dataset_yaml}")

    results = model.train(
        data=dataset_yaml,
        epochs=30,
        imgsz=640,
        batch=32,
        workers=4,
        device=0,
        amp=True,
        project='vision_proyect',
        name=f'augmented_{arch}',
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        patience=10,
        verbose=True,

        # --- Augmentación geométrica ---
        degrees=15.0,        # Rotación ±15°
        translate=0.3,       # Traslación hasta 30% del tamaño
        scale=0.7,           # Escala entre 0.3x y 1.7x
        shear=10.0,          # Shear ±10°
        perspective=0.001,   # Distorsión de perspectiva
        fliplr=0.5,          # Flip horizontal (50%)
        flipud=0.0,          # Sin flip vertical (no tiene sentido en conducción)

        # --- Augmentación de color ---
        hsv_h=0.02,          # Variación de tono
        hsv_s=0.8,           # Variación de saturación
        hsv_v=0.6,           # Variación de brillo (condiciones de luz)

        # --- Augmentación de mezcla ---
        mosaic=1.0,          # Mosaic activo (4 imágenes combinadas)
        mixup=0.15,          # Mezcla suave entre imágenes
        copy_paste=0.1,      # Copia objetos entre imágenes
    )

    print("\n" + "="*30)
    print("Fine-Tuning con Data Augmentation COMPLETADO.")
    print(f"Mejor modelo guardado en: {results.save_dir}")
    print("="*30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8s.pt', help='Modelo base (yolov8s.pt o yolo11s.pt)')
    args = parser.parse_args()
    train_augmented(args.model)
