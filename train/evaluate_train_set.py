import os
from ultralytics import YOLO

def evaluate_train_set():
    # Ruta al mejor modelo Nano entrenado
    model_path = 'runs/detect/vision_proyect/baseline_yolov8n-2/weights/best.pt'
    dataset_yaml = 'trabajo/dataset_yolo/dataset.yaml'
    
    if not os.path.exists(model_path):
        print(f"Error: No se encuentra el modelo en {model_path}")
        return

    model = YOLO(model_path)
    
    print(f"--- Evaluando Modelo en el set de ENTRENAMIENTO ---")
    results = model.val(data=dataset_yaml, split='train', imgsz=640, device=0)
    
    print("\nResultados en TRAIN:")
    print(f"mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision: {results.results_dict['metrics/precision(B)']:.4f}")
    print(f"Recall: {results.results_dict['metrics/recall(B)']:.4f}")

if __name__ == '__main__':
    evaluate_train_set()
