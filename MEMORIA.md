# Memoria — Trabajo Final: Detección de Objetos para Conducción Autónoma
**Redes Neuronales y Computer Vision — UDIT AI Master**

---

## 1. Selección del modelo (5%)

### Modelo elegido: YOLO11s

Se ha seleccionado **YOLO11s** (Ultralytics, octubre 2024) como modelo principal para este trabajo. La elección se justifica por los siguientes motivos:

**Frente a arquitecturas transformer (RF-DETR, RT-DETR):**
- YOLO11s tiene ~9M parámetros frente a los ~32M de RF-DETR Base, lo que permite entrenamiento más ágil y mayor flexibilidad experimental.
- En inferencia alcanza ~150 FPS en una GPU T4, requisito crítico para conducción autónoma en tiempo real. RF-DETR se sitúa en ~60 FPS.
- Los transformers requieren más datos y más epochs para converger. Con un dataset de tamaño moderado (~65K imágenes) y tiempos acotados, YOLO11s ofrece mejor rendimiento práctico.

**Frente a versiones anteriores de YOLO (YOLOv8, YOLOv5):**
- YOLO11 introduce **Dual Label Assignment (DLA)**, que combina asignación de etiquetas dinámica durante el entrenamiento con asignación estática en inferencia, mejorando la convergencia sin coste en velocidad.
- Mejora el bloque **C3k2** respecto al C2f de YOLOv8, aumentando la capacidad representacional del backbone con menos parámetros.
- Mejor mAP en COCO que YOLOv8s con similar número de parámetros (mAP50:95 = 48.0 vs 44.9 en COCO val).

**Adecuación al problema:**
- Las imágenes del dataset son de conducción urbana (nuScenes), con objetos de tamaño variado a distintas distancias. La arquitectura multi-escala de YOLO11 (tres cabezas de detección: P3/P4/P5) cubre bien este rango.
- La resolución de entrada configurable (960×960 en este trabajo) permite aprovechar la resolución original del dataset (960×540) sin degradación severa por redimensionado.

---

## 2. Análisis del modelo (15%)

### 2.1 Arquitectura general

YOLO11 sigue la estructura clásica de los detectores de un solo paso: **backbone → neck → head**.

```
Imagen (960×960)
    │
    ▼
┌─────────────────────────────────────────────┐
│  BACKBONE (extracción de características)    │
│  Conv → C3k2 × N → SPPF                     │
│  Salidas: P3 (120×120), P4 (60×60), P5 (30×30) │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  NECK (FPN + PAN — fusión multi-escala)      │
│  Combina características de los tres niveles │
│  para detectar objetos pequeños, medianos    │
│  y grandes simultáneamente                  │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  HEAD — detección anchor-free                │
│  Tres cabezas independientes (P3, P4, P5)    │
│  Cada celda predice: bbox (4) + obj + clases │
└─────────────────────────────────────────────┘
```

### 2.2 Componentes clave y relación con conceptos de clase

**Bloque C3k2 (backbone):**
Evolución del módulo CSP (Cross-Stage Partial) visto en clase. Divide el tensor de entrada en dos ramas: una pasa por N convoluciones en cascada (aprendizaje profundo) y la otra va directa (residual). Se concatenan al final. Esto permite gradientes más fluidos durante el backpropagation (conexión residual) y mayor riqueza de características con menos parámetros.

**SPPF (Spatial Pyramid Pooling Fast):**
Pooling máximo con tres kernels en cascada, equivalente al SPP tradicional pero más eficiente. Amplía el campo receptivo efectivo de la última capa del backbone sin aumentar la resolución de los feature maps, permitiendo detectar objetos de múltiples escalas.

**FPN + PAN (neck):**
- **FPN** (Feature Pyramid Network): propaga contexto semántico de capas profundas (bajo nivel de detalle, alto nivel semántico) hacia capas superficiales mediante upsampling.
- **PAN** (Path Aggregation Network): el camino inverso, propaga información espacial de capas superficiales (alta resolución) hacia capas profundas.
- El resultado es que cada nivel de la pirámide tiene tanto contexto semántico como detalle espacial, crucial para detectar peatones lejanos y camiones cercanos en la misma imagen.

**Detección anchor-free:**
A diferencia de YOLOv5, YOLO11 no usa anchor boxes predefinidas. Cada celda de la cuadrícula predice directamente offsets respecto a su posición central (Task-Aligned Head). Esto elimina la necesidad de diseñar anchors para el dataset y reduce hiperparámetros.

**Dual Label Assignment (DLA) — novedad YOLO11:**
Durante el entrenamiento se usan dos asignadores en paralelo:
- *Asignador dinámico (TAL)*: asigna ground-truth a las predicciones de mayor calidad según una métrica combinada de clasificación + localización.
- *Asignador estático*: asignación fija por cercanía al centro del objeto.
Los gradientes de ambos se combinan, enriqueciendo la señal de entrenamiento. En inferencia solo opera la cabeza estática, sin overhead.

**Funciones de pérdida:**
- **VariFocal Loss (VFL)**: para clasificación. Asigna más peso a los verdaderos positivos de alta IoU, abordando el desequilibrio entre positivos y negativos (problema severo en nuestro dataset: 144K coches vs 4.6K buses).
- **Distribution Focal Loss (DFL)**: para localización. En vez de predecir un valor único de coordenada, predice una distribución discreta, capturando la incertidumbre inherente en bordes ambiguos.
- **CIoU Loss**: penaliza diferencias en centro, tamaño y ratio de aspecto entre bbox predicho y ground truth.

### 2.3 Hiperparámetros de entrenamiento seleccionados

| Parámetro | Valor | Justificación |
|---|---|---|
| `imgsz` | 960 | Resolución original del dataset (960×540) |
| `batch` | 8 | Límite ~10 GB en GPU A2 16 GB con 960 px |
| `optimizer` | SGD | Mejor generalización final en detección que Adam |
| `lr0` | 0.01 | LR inicial estándar para SGD en YOLO |
| `cos_lr` | True | Cosine annealing: decaimiento suave evita mínimos bruscos |
| `momentum` | 0.937 | Estándar Ultralytics para SGD en detección |
| `weight_decay` | 0.0005 | Regularización L2, evita overfitting en clases mayoritarias |
| `warmup_epochs` | 3 | Estabilización inicial del gradiente |
| `amp` | True | Mixed precision FP16, necesario para caber en VRAM |
| `patience` | 10 | Early stopping si no mejora mAP en 10 epochs |

---

## 3. Conversión de formato COCO a YOLO (15%)

### 3.1 Diferencias entre formatos

**Formato COCO (JSON):**
```json
{
  "images": [{"id": 1, "file_name": "img.jpg", "width": 960, "height": 540}],
  "annotations": [{"image_id": 1, "category_id": 3, "bbox": [x, y, w, h]}],
  "categories": [{"id": 1, "name": "bicycle"}, ...]
}
```
- Un único fichero JSON para todo el split.
- Coordenadas de bbox en píxeles absolutos: `[x_topleft, y_topleft, width, height]`.
- IDs de categoría arbitrarios (no necesariamente 0-based).

**Formato YOLO (TXT por imagen):**
```
# una línea por objeto: class_id cx cy w h  (todo normalizado 0-1)
2 0.512500 0.481481 0.234375 0.296296
```
- Un fichero `.txt` por imagen (misma ruta, distinta carpeta).
- Coordenadas normalizadas por ancho/alto de imagen.
- `class_id` entero desde 0.

**Fórmula de conversión:**
```
cx = (x + w/2) / img_width
cy = (y + h/2) / img_height
w_norm = w / img_width
h_norm = h / img_height
```

### 3.2 Limpieza del dataset original

El dataset original contenía 8 categorías, siendo la primera (`traffic-dataset`, id=0) una categoría vacía sin ninguna anotación. Se eliminó y se renumeraron las 7 clases restantes (1-7 → 0-6) con el script `clean_classes.py`, actualizando tanto los JSON COCO como los ficheros YOLO.

### 3.3 Split estratificado multilabel

El dataset original estaba dividido en Train (81.5%) / Test (18.5%). Para el entrenamiento propio se realizó un nuevo split **80% train / 20% validación** usando `MultilabelStratifiedShuffleSplit` de la librería `iterative-stratification`.

Esta técnica garantiza que la distribución de cada clase se preserve en ambos splits, lo cual es crítico dado el fuerte desequilibrio del dataset:

| Clase | Total | Train | Val | Val% |
|---|---:|---:|---:|---:|
| bicycle | 11.000 | 8.791 | 2.209 | 20,1% |
| bus | 4.613 | 3.682 | 931 | 20,2% |
| car | 144.379 | 115.311 | 29.068 | 20,1% |
| human | 105.189 | 84.528 | 20.661 | 19,6% |
| motorcycle | 10.346 | 8.252 | 2.094 | 20,2% |
| trafficcone | 52.155 | 41.762 | 10.393 | 19,9% |
| truck | 23.788 | 19.041 | 4.747 | 20,0% |

El dataset presenta un **desequilibrio severo**: `car` (41% de las instancias) tiene 31× más ejemplos que `bus` (1,3%). Esto motiva directamente el experimento de data augmentation descrito en el apartado 4.

*[Insertar aquí: gráfico class_dist_total.png y class_dist_pct.png]*

---

## 4. Entrenamiento con múltiples técnicas (32.5%)

Se han diseñado cuatro experimentos sobre el mismo modelo base (YOLO11s, 960px) para aislar el efecto de cada técnica. Todos comparten los mismos hiperparámetros base (SGD, lr0=0.01, cos_lr, batch=8) y solo difieren en la técnica aplicada.

### Experimento 1 — Finetuning base (sin augmentación)

**Técnica:** Fine-tuning completo desde pesos preentrenados en COCO (ImageNet backbone + detección COCO). Todos los pesos son actualizables.  
**Augmentación:** Ninguna (toda desactivada a 0).  
**Objetivo:** Establecer el baseline del modelo para medir el efecto de cada técnica por separado.  
**Script:** `compvision/scripts/train_yolo11s_960.py --hyp hyp_yolo11s_960_base.yaml`

| Métrica | Valor |
|---|---|
| mAP 50:95 | [PENDIENTE] |
| mAP 50 | [PENDIENTE] |
| mAP 75 | [PENDIENTE] |
| Precision | [PENDIENTE] |
| Recall | [PENDIENTE] |
| Epochs hasta convergencia | [PENDIENTE] |

*[Insertar curvas de loss y mAP por epoch]*

---

### Experimento 2 — Finetuning con data augmentation

**Técnica:** Fine-tuning completo con augmentación agresiva, especialmente diseñada para compensar el desequilibrio de clases.  
**Augmentación activada:**
- `mosaic=1.0`: combina 4 imágenes en una sola, multiplicando efectivamente la variedad de contextos para clases minoritarias (bus, bicycle, motorcycle).
- `copy_paste=0.3`: copia objetos de otras imágenes y los pega en la imagen actual. Especialmente útil para bus (4.613 instancias) y motorcycle (10.346).
- `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4`: variación de color para simular diferentes condiciones de iluminación (día/noche/lluvia).
- `fliplr=0.5`: flip horizontal (válido en conducción).
- `scale=0.5`: zoom aleatorio ±50%, simula objetos a diferentes distancias.

**Objetivo:** Medir cuánto mejoran las clases minoritarias respecto al baseline.  
**Script:** `compvision/scripts/train_yolo11s_960.py --hyp hyp_yolo11s_960_aug.yaml`

| Métrica | Valor |
|---|---|
| mAP 50:95 | [PENDIENTE] |
| mAP 50 | [PENDIENTE] |
| AP bus | [PENDIENTE] |
| AP bicycle | [PENDIENTE] |
| AP motorcycle | [PENDIENTE] |
| Epochs hasta convergencia | [PENDIENTE] |

---

### Experimento 3 — Transfer learning (backbone congelado)

**Técnica:** Solo se entrenan las capas del neck y la cabeza de detección. El backbone (extractor de características) mantiene los pesos de COCO congelados.  
**Justificación:** Útil cuando el dataset es pequeño o el dominio es similar al de preentrenamiento. Permite adaptar el detector al vocabulario de 7 clases sin olvidar las características visuales genéricas aprendidas.  
**Layers congeladas:** `freeze=10` (primeros 10 módulos del backbone).

| Métrica | Valor |
|---|---|
| mAP 50:95 | [PENDIENTE] |
| mAP 50 | [PENDIENTE] |
| Precision | [PENDIENTE] |
| Recall | [PENDIENTE] |
| Epochs hasta convergencia | [PENDIENTE] |

**Análisis esperado:** El transfer learning debería converger más rápido (menos parámetros libres) pero alcanzar un mAP final inferior al finetuning completo, dado que el backbone no se adapta al dominio específico de conducción nocturna/lluvia del dataset nuScenes.

---

### Experimento 4 — Entrenamiento from scratch

**Técnica:** Inicialización aleatoria de todos los pesos (sin preentrenamiento). Permite medir el valor del transfer learning comparando directamente con los experimentos 1 y 3.  
**Parámetro:** `pretrained=False` en la carga del modelo.  
**Observación esperada:** Convergencia más lenta, mAP final inferior, mayor dependencia del número de epochs y la tasa de aprendizaje.

| Métrica | Valor |
|---|---|
| mAP 50:95 | [PENDIENTE] |
| mAP 50 | [PENDIENTE] |
| Precision | [PENDIENTE] |
| Recall | [PENDIENTE] |
| Epochs hasta convergencia | [PENDIENTE] |

---

## 5. Presentación y análisis de resultados (15%)

### 5.1 Tabla comparativa global

| Experimento | mAP 50:95 | mAP 50 | Precision | Recall | Epochs |
|---|---|---|---|---|---|
| E1 — Finetuning base | [P] | [P] | [P] | [P] | [P] |
| E2 — + Augmentation | [P] | [P] | [P] | [P] | [P] |
| E3 — Backbone frozen | [P] | [P] | [P] | [P] | [P] |
| E4 — From scratch | [P] | [P] | [P] | [P] | [P] |

### 5.2 Resultados por clase — Mejor modelo

| Clase | AP 50:95 | AP 50 | Precision | Recall |
|---|---|---|---|---|
| bicycle | [P] | [P] | [P] | [P] |
| bus | [P] | [P] | [P] | [P] |
| car | [P] | [P] | [P] | [P] |
| human | [P] | [P] | [P] | [P] |
| motorcycle | [P] | [P] | [P] | [P] |
| trafficcone | [P] | [P] | [P] | [P] |
| truck | [P] | [P] | [P] | [P] |

### 5.3 Análisis de métricas COCO

Las métricas usadas son las del reto de detección COCO:

- **mAP@0.5 (AP50):** IoU threshold = 0.50. Métrica más permisiva, mide si el modelo localiza correctamente los objetos aunque el bbox no sea preciso.
- **mAP@0.5:0.95 (AP50:95):** Promedio de AP a 10 umbrales IoU (0.50, 0.55, ..., 0.95). Métrica principal COCO. Penaliza localizaciones imprecisas. Es la más exigente y la que mejor discrimina entre modelos.
- **Precision:** De todas las detecciones realizadas, qué % son correctas. Alta precision = pocos falsos positivos.
- **Recall:** De todos los objetos presentes, qué % se detectaron. Alto recall = pocos falsos negativos.

**Relación entre métricas y clases problemáticas:**
- `bus` y `motorcycle` esperamos que tengan el AP más bajo en E1 por su escasez de instancias. E2 debería mejorarlos notablemente.
- `car` y `human` tienen AP alto de base por volumen de entrenamiento, pero pueden sufrir si el modelo les da demasiado peso (alta precision, bajo recall en las minoritarias).
- `trafficcone` puede beneficiarse especialmente del mosaic, al ser objetos pequeños que aparecen en grupos.

*[Insertar curvas PR por clase del mejor modelo]*
*[Insertar curvas de training loss (box, cls, dfl) de los 4 experimentos]*

---

## 6. Tracking (10%)

Se añade **ByteTrack** como algoritmo de tracking multi-objeto sobre el modelo YOLO11s.

**Justificación de ByteTrack frente a alternativas:**
- ByteTrack no requiere un modelo de re-identificación (ReID) separado, reduciendo la latencia total.
- Usa todas las detecciones (incluyendo las de baja confianza) para mantener trayectorias en situaciones de oclusión parcial, frecuentes en conducción urbana.
- Integrado nativamente en Ultralytics con `model.track()`.

**Principio de funcionamiento:**
1. El detector genera bboxes + scores en cada frame.
2. Los objetos de alta confianza se asocian a trayectorias existentes usando distancia IoU (Hungarian algorithm).
3. Los objetos de baja confianza se usan para recuperar trayectorias perdidas (no se descartan como en SORT).
4. Cada objeto mantiene un `track_id` persistente entre frames.

**Configuración:**
```python
model.track(
    source=video_path,
    tracker="bytetrack.yaml",
    conf=0.3,       # threshold detección [PENDIENTE: ajustar]
    iou=0.5,        # threshold NMS
    persist=True,   # mantiene IDs entre llamadas
)
```

---

## 7. Ejecución sobre video (7.5%)

**Video utilizado:** [PENDIENTE: nombre del video proporcionado]  
**Hardware:** NVIDIA A2 16 GB  

**Thresholds finales seleccionados:**

| Parámetro | Valor | Criterio |
|---|---|---|
| `conf` (detección) | [PENDIENTE] | Equilibrio FP/FN en el video |
| `iou` (NMS) | [PENDIENTE] | Sin detecciones duplicadas en objetos cercanos |
| `tracker conf` | [PENDIENTE] | Estabilidad de IDs sin fragmentación |

**Proceso de ajuste de thresholds:**
Se probaron valores de `conf` entre 0.20 y 0.50 sobre el video. Un threshold bajo aumenta el recall (se detectan más objetos lejanos/parciales) pero introduce falsos positivos en fondos. El valor elegido maximiza las detecciones útiles para conducción (peatones, vehículos) minimizando las detecciones de ruido.

*[Insertar capturas del video con detecciones + tracking]*
*[Insertar métricas de velocidad: FPS medio en inferencia]*

---

## Conclusiones

[PENDIENTE tras completar todos los experimentos]

Puntos clave a desarrollar:
- Comparativa finetuning vs transfer learning: ¿cuánto importa el preentrenamiento?
- Impacto del data augmentation en clases minoritarias (bus, motorcycle, bicycle).
- Trade-off velocidad/precisión para conducción autónoma en tiempo real.
- Limitaciones del dataset: desequilibrio de clases, condiciones de iluminación.

---

## Apéndice — Infraestructura y reproducibilidad

**Hardware:** 4× NVIDIA A2 16 GB | RF-DETR en GPU3 | YOLO11s en GPU2  
**Tracking de experimentos:** MLflow en http://localhost:5050  
**Repositorio:** https://github.com/luisgs69/udit-sdc-vision  

**Reproducción completa:**
```bash
git clone https://github.com/luisgs69/udit-sdc-vision
pip install -r requirements.txt

# 1. Preparar dataset
python3 compvision/scripts/coco_to_yolo.py --coco_json train/labels.json
python3 compvision/scripts/clean_classes.py

# 2. Entrenamiento (elegir experimento)
python3 compvision/scripts/train_yolo11s_960.py --hyp hyp_yolo11s_960_base.yaml

# 3. Visualizar métricas
mlflow ui --port 5050
```
