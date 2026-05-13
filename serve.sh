#!/bin/bash
# Uso: ./serve.sh <modelo> <entrenamiento> [puerto]
# Ejemplos:
#   ./serve.sh yolov8s finetuning
#   ./serve.sh yolo11s frozen 8001
#   ./serve.sh rtdetr_l finetuning 8002

MODEL=$1
TRAINING=$2
PORT=${3:-8000}

if [ -z "$MODEL" ] || [ -z "$TRAINING" ]; then
    echo "Uso: ./serve.sh <modelo> <entrenamiento> [puerto]"
    echo "  Modelos:      yolov8s | yolo11s | rtdetr_l | yolov8n"
    echo "  Entrenamientos: finetuning | frozen | augmented | baseline"
    exit 1
fi

# Mapeo modelo+entrenamiento → carpeta en runs/
declare -A RUN_DIRS
RUN_DIRS["yolov8s_finetuning"]="finetuning_yolov8s_opt"
RUN_DIRS["yolov8s_frozen"]="frozen_yolov8s"
RUN_DIRS["yolov8s_augmented"]="augmented_yolov8s"
RUN_DIRS["yolo11s_finetuning"]="finetuning_yolo11s"
RUN_DIRS["yolo11s_frozen"]="frozen_yolo11s"
RUN_DIRS["yolo11s_augmented"]="augmented_yolo11s"
RUN_DIRS["rtdetr_l_finetuning"]="finetuning_rtdetr_l"
RUN_DIRS["yolov8n_baseline"]="baseline_yolov8n"

KEY="${MODEL}_${TRAINING}"
RUN_DIR=${RUN_DIRS[$KEY]}

if [ -z "$RUN_DIR" ]; then
    echo "Error: combinación '$KEY' no reconocida."
    echo "Combinaciones válidas: ${!RUN_DIRS[@]}"
    exit 1
fi

WEIGHTS="/ultralytics/runs/detect/vision_proyect/${RUN_DIR}/weights/best.pt"
CONTAINER_NAME="api_${MODEL}_${TRAINING}"

# Parar contenedor anterior con el mismo nombre si existe
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Parando contenedor existente: ${CONTAINER_NAME}"
    docker rm -f "$CONTAINER_NAME"
fi

echo "Levantando API:"
echo "  Contenedor : $CONTAINER_NAME"
echo "  Modelo     : $WEIGHTS"
echo "  Puerto     : $PORT"

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus '"device=3"' \
    -p "${PORT}:8000" \
    -e MODEL_PATH="$WEIGHTS" \
    -v "$(pwd)/runs:/ultralytics/runs" \
    modelos

echo ""
echo "API disponible en http://localhost:${PORT}"
echo "  Health : curl http://localhost:${PORT}/health"
echo "  Detect : curl -X POST http://localhost:${PORT}/detect -F 'file=@imagen.jpg'"
