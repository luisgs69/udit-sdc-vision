# SDC Vision — Detección de Objetos para Conducción Autónoma

Trabajo Final — Redes Neuronales y Computer Vision (UDIT AI Master)

Detección de objetos en entornos 2D para conducción autónoma sobre el dataset nuScenes (7 clases: bicycle, bus, car, human, motorcycle, trafficcone, truck).

## Modelos implementados

| Modelo | Script | Técnica |
|---|---|---|
| YOLOv8s | `train/train_finetuning.py` | Finetuning completo |
| YOLOv8s | `train/train_frozen.py` | Transfer learning (backbone congelado) |
| YOLOv8s | `train/train_augmented.py` | Finetuning + data augmentation |
| YOLOv8n | `train/train_baseline.py` | Baseline desde preentrenado |
| YOLO11s | `train/train_yolo11.py` | Finetuning |
| RT-DETR L | `train/train_rtdetr.py` | Finetuning transformer |
| RF-DETR Base | `train/train_rfdetr.py` | Finetuning DINOv2 |

## Requisitos

- Docker con soporte GPU (NVIDIA Container Toolkit)
- Dataset en formato COCO (`dataset_coco/`) y YOLO (`dataset_yolo/`)
- El dataset NO está incluido en este repo por su tamaño (~7GB)

## Setup rápido

```bash
# 1. Construir imagen Docker
docker build -t sdc-vision .

# 2. Arrancar MLflow (métricas en tiempo real)
./mlflow_server.sh
# UI disponible en http://localhost:5050

# 3. Lanzar entrenamiento (ejemplo: finetuning YOLOv8s)
docker run --gpus '"device=0"' \
  -v $(pwd)/dataset_yolo:/usr/src/app/dataset_yolo \
  -v $(pwd)/runs:/ultralytics/runs \
  sdc-vision python3 train/train_finetuning.py

# 4. Lanzar API de inferencia
./serve.sh yolov8s finetuning
```

## Dataset

Formato COCO → YOLO con estratificación multilabel:

```bash
# Conversión y split 80/20 estratificado por clase
python3 compvision/scripts/coco_to_yolo.py

# Verificar distribución
python3 compvision/scripts/get_yolo_class_distribution.py --dataset_dir dataset_yolo

# Preparar dataset COCO para RF-DETR
python3 prepare_coco_rfdetr.py
```

## Auditoría de etiquetas

```bash
python3 check_labels.py --split train --dataset_dir dataset_yolo
python3 fix_labels.py
```

## Evaluación

```bash
# Comparativa multi-modelo con métricas COCO
python3 train/evaluate_models.py

# Métricas COCO estándar (AP por IoU, AP por clase)
python3 evaluate_coco_metrics.py
```

## Estructura

```
├── train/                  # Scripts de entrenamiento
│   ├── mlflow_setup.py     # Config MLflow compartida
│   ├── train_*.py          # Un script por modelo/técnica
│   └── evaluate_*.py       # Evaluación y comparativa
├── compvision/scripts/     # Utilidades dataset
├── api/main.py             # API FastAPI inferencia
├── dataset_yolo/
│   └── dataset.yaml        # Config clases y rutas
├── Dockerfile
├── requirements.txt
├── serve.sh                # Lanzar API por modelo
└── mlflow_server.sh        # Arrancar servidor MLflow
```
