# Resultados — Detección de Objetos en Conducción Autónoma

## 1. Dataset

### Formato y split global

| Split | Imágenes | Anotaciones | Imágenes sin anotación |
|---|---|---|---|
| Train | 42.280 | 281.367 | 5.275 (12,5%) |
| Val | 10.571 | 70.103 | 1.333 (12,6%) |
| **Total** | **52.851** | **351.470** | **6.608 (12,5%)** |

Split ~80/20 estratificado por clase. Ambos formatos (YOLO y COCO) comparten las mismas imágenes y anotaciones; la diferencia es únicamente la representación.

---

### 1.1 Estratificación — Formato YOLO

Generado con `compvision/scripts/get_yolo_class_distribution.py`. Clases referenciadas por ID numérico en los `.txt`.

| ID | Clase | Total | Train | Val | Train % | Val % |
|---|---|---|---|---|---|---|
| 0 | bicycle | 11.000 | 8.791 | 2.209 | 79,92% | 20,08% |
| 1 | bus | 4.613 | 3.682 | 931 | 79,82% | 20,18% |
| 2 | car | 144.379 | 115.311 | 29.068 | 79,87% | 20,13% |
| 3 | human | 105.189 | 84.528 | 20.661 | 80,36% | 19,64% |
| 4 | motorcycle | 10.346 | 8.252 | 2.094 | 79,76% | 20,24% |
| 5 | trafficcone | 52.155 | 41.762 | 10.393 | 80,07% | 19,93% |
| 6 | truck | 23.788 | 19.041 | 4.747 | 80,04% | 19,96% |
| — | **TOTAL** | **351.470** | **281.367** | **70.103** | **80,05%** | **19,95%** |

Estructura: `dataset_yolo/labels/{train,val}/*.txt` — una línea por instancia: `<class_id> <xc> <yc> <w> <h>` (coordenadas normalizadas).

---

### 1.2 Estratificación — Formato COCO

Generado desde `dataset_coco/{train,valid}/_annotations.coco.json`. Clases referenciadas por `category_id` en el JSON.

| Clase | Total | Train | % train | Val | % val | Ratio train/val |
|---|---|---|---|---|---|---|
| bicycle | 11.000 | 8.791 | 3,1% | 2.209 | 3,2% | 79,9% / 20,1% |
| bus | 4.613 | 3.682 | 1,3% | 931 | 1,3% | 79,8% / 20,2% |
| car | 144.379 | 115.311 | 41,0% | 29.068 | 41,5% | 79,9% / 20,1% |
| human | 105.189 | 84.528 | 30,0% | 20.661 | 29,5% | 80,4% / 19,6% |
| motorcycle | 10.346 | 8.252 | 2,9% | 2.094 | 3,0% | 79,8% / 20,2% |
| trafficcone | 52.155 | 41.762 | 14,8% | 10.393 | 14,8% | 80,1% / 19,9% |
| truck | 23.788 | 19.041 | 6,8% | 4.747 | 6,8% | 80,0% / 20,0% |
| **TOTAL** | **351.470** | **281.367** | **100%** | **70.103** | **100%** | **80,0% / 20,0%** |

Estructura: JSON con campos `images`, `annotations` (con `bbox` en formato `[x, y, w, h]` absoluto) y `categories`.

---

### Observaciones sobre la estratificación

- El split 80/20 se mantiene con una desviación máxima de **±0,6 pp** en cualquier clase.
- `car` (41%) y `human` (30%) dominan el dataset; `bus` (1,3%) es la clase más escasa.
- El 12,5% de imágenes carece de anotación en ambos splits (imágenes de fondo), porcentaje idéntico en train y val.

---

## 2. Métricas de validación por modelo

### 2.1 YOLOv8s (finetuning_yolov8s_opt) — 30 epochs completados

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 1 | — | — | — | — |
| 10 | 0,800 | 0,582 | 0,680 | 0,434 |
| 20 | 0,792 | 0,597 | 0,688 | 0,441 |
| 25 | 0,793 | 0,600 | 0,690 | 0,445 |
| **30 (final)** | **0,787** | **0,604** | **0,692** | **0,448** |

Ruta resultados: `runs/detect/vision_proyect/finetuning_yolov8s_opt/results.csv`

### 2.2 RT-DETR L (finetuning_rtdetr_l-2) — 10 epochs (interrumpido)

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 1 | 0,570 | 0,239 | 0,218 | 0,099 |
| 3 | 0,699 | 0,524 | 0,590 | 0,361 |
| 5 | 0,757 | 0,595 | 0,680 | 0,428 |
| 7 | 0,780 | 0,625 | 0,712 | 0,453 |
| 9 | 0,788 | 0,642 | 0,731 | 0,468 |
| **10 (último)** | **0,792** | **0,650** | **0,737** | **0,474** |

Ruta resultados: `runs/detect/vision_proyect/finetuning_rtdetr_l-2/results.csv`

### 2.3 RF-DETR base (finetuning_rfdetr_base) — EN CURSO (30 epochs total)

| Epoch | Precision | Recall | mAP50 | mAP50-95 | ema_mAP50 |
|---|---|---|---|---|---|
| 3 | 0,773 | 0,589 | 0,677 | 0,422 | 0,683 |
| 5 | 0,782 | 0,595 | 0,690 | 0,430 | 0,694 |
| 7 | 0,774 | 0,607 | 0,692 | 0,432 | 0,701 |
| 9 | 0,781 | 0,614 | 0,701 | 0,438 | 0,707 |
| 10 | 0,785 | 0,614 | 0,704 | 0,443 | 0,710 |
| **11 (último val)** | **0,772** | **0,624** | **0,706** | **0,445** | **0,712** |

- Estado: epoch 12/30 en curso (actualizado 2026-05-13)
- Checkpoints: `runs/detect/vision_proyect/finetuning_rfdetr_base/`
- Mejor checkpoint EMA: `checkpoint_best_ema.pth`

---

## 3. Comparativa resumida (mejor epoch de cada modelo)

| Modelo | Epochs | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| YOLOv8s | 30 | 0,692 | 0,448 | 0,787 | 0,604 |
| RT-DETR L | 10 | 0,737 | 0,474 | 0,792 | 0,650 |
| RF-DETR base | 11* | 0,706 | 0,445 | 0,772 | 0,624 |

*RF-DETR base aún en entrenamiento; valores provisionales.

RT-DETR L lidera en mAP50 y recall con solo 10 epochs. RF-DETR base es prometedor: supera a YOLOv8s en recall (+2 pp) con la mitad de epochs entrenados.

---

## 4. Estructura de directorios

```
trabajo/
│
├── dataset_coco/                        # Dataset en formato COCO (para RF-DETR)
│   │                                    # No tiene YAML: rfdetr lee el directorio
│   │                                    # directamente; los JSON son el equivalente
│   │                                    # al dataset.yaml de YOLO
│   ├── train/
│   │   ├── _annotations.coco.json       # 42.280 imgs · 281.367 anns · 7 clases
│   │   └── *.jpg                        # Imágenes train (mismo set que dataset_yolo)
│   └── valid/
│       ├── _annotations.coco.json       # 10.571 imgs · 70.103 anns · 7 clases
│       └── *.jpg                        # Imágenes val
│
├── dataset_yolo/                        # Dataset en formato YOLO (para YOLOv8/RT-DETR)
│   ├── dataset.yaml                     # Config: rutas train/val + 7 nombres de clase
│   ├── images/
│   │   ├── train/                       # 42.280 imágenes JPG
│   │   └── val/                         # 10.571 imágenes JPG
│   ├── labels/
│   │   ├── train/                       # 42.280 .txt  <class_id xc yc w h>
│   │   └── val/                         # 10.571 .txt
│   └── labels_backup/                   # Copia pre-fix_labels.py
│       └── train/
│
├── label_check/                         # Auditoría de calidad de etiquetas
│   ├── grid_samples_train.jpg           # Grid de muestras aleatorias train
│   ├── grid_samples_val.jpg             # Grid de muestras aleatorias val
│   ├── grid_suspicious_train.jpg        # Anotaciones atípicas train
│   ├── grid_suspicious_val.jpg          # Anotaciones atípicas val
│   ├── label_issues_train.txt           # Lista detallada de issues train
│   └── label_issues_val.txt             # Lista detallada de issues val
│
├── runs/detect/vision_proyect/          # Resultados de entrenamiento
│   ├── finetuning_rfdetr_base/          # RF-DETR base — EN CURSO (epoch 12/30)
│   │   ├── metrics.csv                  # Métricas step a step + val por epoch
│   │   ├── checkpoint_best_ema.pth      # Mejor checkpoint (EMA)
│   │   ├── checkpoint_best_regular.pth  # Mejor checkpoint (regular)
│   │   ├── last.ckpt                    # Último checkpoint guardado
│   │   └── events.out.tfevents.*        # Logs TensorBoard
│   ├── finetuning_rtdetr_l-2/           # RT-DETR L — 10 epochs (interrumpido)
│   │   ├── results.csv                  # Métricas epoch a epoch
│   │   ├── weights/                     # best.pt + last.pt
│   │   ├── args.yaml                    # Configuración del run
│   │   ├── labels.jpg                   # Distribución de labels
│   │   └── train_batch*.jpg             # Muestras de batches
│   ├── finetuning_rtdetr_l/             # RT-DETR L — primera ejecución (ref.)
│   │   └── results.csv
│   └── finetuning_yolov8s_opt/          # YOLOv8s — 30 epochs completos
│       ├── results.csv                  # Métricas epoch a epoch
│       ├── results.png                  # Curvas de entrenamiento
│       ├── confusion_matrix.png         # Matriz de confusión
│       ├── confusion_matrix_normalized.png
│       ├── Box{F1,P,R,PR}_curve.png     # Curvas F1, Precision, Recall, PR
│       ├── val_batch*_labels.jpg        # Ground truth validación
│       ├── val_batch*_pred.jpg          # Predicciones validación
│       └── weights/                     # best.pt + last.pt
│
├── train/                               # Scripts de entrenamiento
│   ├── data/                            # Imágenes para RF-DETR (symlink/copia)
│   ├── labels.json                      # Mapeo id→nombre para RF-DETR
│   ├── train_baseline.py                # Baseline YOLOv8 sin modificar
│   ├── train_augmented.py               # YOLOv8 con augmentation extra
│   ├── train_finetuning.py              # Finetuning YOLOv8 (capas descongeladas)
│   ├── train_frozen.py                  # YOLOv8 con backbone congelado
│   ├── train_rtdetr.py                  # Entrenamiento RT-DETR L
│   ├── train_rfdetr.py                  # Entrenamiento RF-DETR base (activo)
│   ├── train_yolo11.py                  # YOLO11 (experimental)
│   ├── evaluate_models.py               # Evaluación comparativa multi-modelo
│   └── evaluate_train_set.py            # Evaluación del modelo en set train
│
├── compvision/                          # Código base del módulo de visión
│   ├── scripts/
│   │   ├── coco_to_yolo.py              # Conversión anotaciones COCO→YOLO
│   │   ├── get_yolo_class_distribution.py  # Estratificación por clase YOLO
│   │   └── visualize_dataset.py         # Visualización de imágenes + bboxes
│   ├── metricas.pdf                     # PDF con métricas de referencia
│   ├── README.md                        # Descripción del proyecto
│   └── requirements.txt
│
├── api/
│   └── main.py                          # API FastAPI para inferencia
│
├── check_labels.py                      # Auditoría etiquetas YOLO (genera label_check/)
├── fix_labels.py                        # Corrección de etiquetas problemáticas
├── evaluate_coco_metrics.py             # Evaluación con métricas estándar COCO
├── prepare_coco_rfdetr.py               # Adaptación dataset COCO para RF-DETR
│
├── INSTRUCCIONES_SERVIDOR.md            # Setup entorno GPU servidor
├── RESULTADOS.md                        # Este documento
├── memoria_final.tex / .pdf             # Memoria del proyecto (LaTeX)
├── Dockerfile                           # Container para inferencia en producción
├── requirements.txt                     # Dependencias Python del proyecto
├── serve.sh                             # Script arranque API de inferencia
└── yolov8s.pt                           # Pesos base YOLOv8s (pretrained COCO)
```

---

## 5. Archivos clave de resultados

| Archivo | Descripción |
|---|---|
| `dataset_coco/train/_annotations.coco.json` | Anotaciones COCO train |
| `dataset_coco/valid/_annotations.coco.json` | Anotaciones COCO val |
| `dataset_yolo/dataset.yaml` | Config YOLO (rutas + 7 clases) |
| `runs/.../finetuning_yolov8s_opt/results.csv` | Métricas YOLOv8s epoch a epoch |
| `runs/.../finetuning_rtdetr_l-2/results.csv` | Métricas RT-DETR L epoch a epoch |
| `runs/.../finetuning_rfdetr_base/metrics.csv` | Métricas RF-DETR base (en curso) |
| `evaluate_coco_metrics.py` | Script evaluación métricas COCO estándar |
| `compvision/scripts/get_yolo_class_distribution.py` | Script estratificación YOLO |
| `label_check/` | Grids y listas de etiquetas sospechosas |
