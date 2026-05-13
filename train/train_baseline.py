import os
from ultralytics import YOLO
from mlflow_setup import setup as mlflow_setup

def train_baseline():
    mlflow_setup()
    # Cargar el modelo YOLOv8 nano pre-entrenado
    model = YOLO('yolov8n.pt')

    # Ruta al archivo de configuración del dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_yaml = os.path.abspath(os.path.join(script_dir, '..', 'dataset_yolo', 'dataset.yaml'))

    print(f"Iniciando entrenamiento base con GPU en: {dataset_yaml}")

    # Entrenamiento
    results = model.train(
        data=dataset_yaml,
        epochs=5,
        imgsz=640,
        batch=16,
        device=0,  # Usar la GPU RTX 2000
        project='vision_proyect',
        name='baseline_yolov8n',
        verbose=True
    )

    print("Entrenamiento base completado.")
    print(f"Resultados guardados en: {results.save_dir}")

if __name__ == '__main__':
    train_baseline()
