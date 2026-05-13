# Guía de Entrenamiento en Servidor Remoto (NVIDIA A2)

Este documento detalla los pasos para ejecutar el entrenamiento de YOLOv8 en un servidor Ubuntu con 4 GPUs, utilizando específicamente la **GPU 3** (16GB VRAM).

## 1. Preparación en local
Comprimir los archivos necesarios (excluyendo datos temporales y entornos locales):

```powershell
# En PowerShell (Windows)
Compress-Archive -Path "trabajo", "yolov8s.pt", "yolov8n.pt" -DestinationPath "proyecto_vision.zip" -Force
```

## 2. Despliegue en el Servidor Ubuntu
Una vez subido el archivo `proyecto_vision.zip` al servidor:

1. **Descomprimir**:
   ```bash
   unzip proyecto_vision.zip
   cd proyecto_vision
   ```

2. **Construir la imagen de Docker**:
   ```bash
   docker build -t yolo_a2_train -f trabajo/Dockerfile .
   ```

## 3. Ejecución del Entrenamiento (GPU 3)
Para lanzar el entrenamiento específicamente (sobrescribiendo el CMD de la API):

```bash
docker run -d \
    --name training_vision_final \
    --gpus '"device=3"' \
    -v $(pwd)/runs:/usr/src/app/runs \
    yolo_a2_train \
    python3 train/train_finetuning.py
```

### Parámetros clave utilizados:
- **Batch Size**: 32 (Optimizado para los 16GB de la A2).
- **Modelo**: YOLOv8s (Small) para mayor precisión que el Nano.
- **Persistencia**: Los pesos y gráficas se guardarán en la carpeta `./runs/` del servidor gracias al volumen montado.

## 4. Monitorización y Control

*   **Ver progreso en tiempo real**:
    ```bash
    docker logs -f training_vision_final
    ```
*   **Comprobar uso de GPU**:
    ```bash
    nvidia-smi
    ```
*   **Detener entrenamiento**:
    ```bash
    docker stop training_vision_final
    ```

## 5. Recuperación de Resultados
Una vez finalizado el entrenamiento (aprox. 3-4 horas), descarga la carpeta `runs/detect/vision_proyect/finetuning_yolov8s_opt/weights/best.pt` para realizar la evaluación y el tracking localmente.
