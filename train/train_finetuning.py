import os
from ultralytics import YOLO
from mlflow_setup import setup as mlflow_setup

def train_finetuning():
    mlflow_setup()
    # Cargamos la versión 'small' para el Fine-tuning (más precisa que la nano)
    model = YOLO('yolov8s.pt')

    # Ruta absoluta al dataset.yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_yaml = os.path.abspath(os.path.join(script_dir, '..', 'dataset_yolo', 'dataset.yaml'))

    print(f"Iniciando Fine-Tuning Optimizado en GPU: {dataset_yaml}")
    print("Progreso: El entrenamiento consta de 30 épocas. Podrás ver el avance detallado abajo.")

    # Entrenamiento con optimizaciones para NVIDIA A2 (16GB VRAM)
    results = model.train(
        data=dataset_yaml,
        epochs=30,           
        imgsz=640,
        batch=32,            # Optimizado para 16GB VRAM
        workers=4,           # Evita cuellos de botella en CPUs de servidor
        device=0,            # El contenedor verá la GPU asignada como device 0
        amp=True,            
        project='vision_proyect',
        name='finetuning_yolov8s_opt',
        optimizer='AdamW',   
        lr0=0.001,           
        cos_lr=True,         
        patience=10,         
        verbose=True
    )

    print("\n" + "="*30)
    print("Fine-tuning COMPLETADO.")
    print(f"Mejor modelo guardado en: {results.save_dir}")
    print("="*30)

if __name__ == '__main__':
    train_finetuning()
