import gc
import io
import os
import tempfile
import threading

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from ultralytics import YOLO

app = FastAPI(
    title="Vision Autonomous Driving API",
    description="API de detección de objetos con gestión dinámica de modelos (un modelo en VRAM a la vez)",
)

RUNS_BASE    = "/ultralytics/runs/detect/vision_proyect"
MODELOS_BASE = "/modelos"

# Modelos Ultralytics → .pt | Modelos RF-DETR → .pth
MODEL_REGISTRY = {
    # ── Servidor: Ultralytics ─────────────────────────────────────────
    "yolov8n_baseline":      f"{RUNS_BASE}/baseline_yolov8n/weights/best.pt",
    "yolov8s_finetuning":    f"{RUNS_BASE}/finetuning_yolov8s_opt/weights/best.pt",
    "yolov8s_frozen":        f"{RUNS_BASE}/frozen_yolov8s/weights/best.pt",
    "yolov8s_augmented":     f"{RUNS_BASE}/augmented_yolov8s/weights/best.pt",
    "yolo11s_finetuning":    f"{RUNS_BASE}/finetuning_yolo11s/weights/best.pt",
    "yolo11s_frozen":        f"{RUNS_BASE}/frozen_yolo11s/weights/best.pt",
    "yolo11s_augmented":     f"{RUNS_BASE}/augmented_yolo11s/weights/best.pt",
    "rtdetr_l_finetuning":   f"{RUNS_BASE}/finetuning_rtdetr_l/weights/best.pt",
    # ── Servidor: RF-DETR ─────────────────────────────────────────────
    "rfdetr_base_finetuning": f"{RUNS_BASE}/finetuning_rfdetr_base/best.pth",
    # ── Portátil: directorio modelos/ ────────────────────────────────
    "yolo8n_entrenado":      f"{MODELOS_BASE}/yolo8nano_entrenado.pt",
    "yolo8s_entrenado":      f"{MODELOS_BASE}/yolo8small_entrenado.pt",
}

RFDETR_MODELS = {"rfdetr_base_finetuning"}

CLASSES = ["bicycle", "bus", "car", "human", "motorcycle", "trafficcone", "truck"]

# Estado global — un único modelo cargado en VRAM
state = {
    "model":      None,
    "model_name": None,
    "model_type": None,   # "ultralytics" | "rfdetr"
}
_lock = threading.Lock()


def _unload_current():
    if state["model"] is not None:
        del state["model"]
        state["model"]      = None
        state["model_name"] = None
        state["model_type"] = None
        gc.collect()
        torch.cuda.empty_cache()


def _load_model(name: str):
    path = MODEL_REGISTRY.get(name)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{name}' no encontrado. Disponibles: {list(MODEL_REGISTRY)}",
        )
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"Weights no encontrados en {path}. ¿Ha terminado el entrenamiento?",
        )

    _unload_current()

    if name in RFDETR_MODELS:
        from rfdetr import RFDETRBase
        model = RFDETRBase(pretrain_weights=path)
        state["model_type"] = "rfdetr"
    else:
        model = YOLO(path)
        state["model_type"] = "ultralytics"

    state["model"]      = model
    state["model_name"] = name
    print(f"Modelo cargado: {name} ({state['model_type']})")


def _detect_ultralytics(image: Image.Image) -> list[dict]:
    img_array = np.array(image)
    results = state["model"](img_array)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id   = int(box.cls[0].item())
            cls_name = state["model"].names.get(cls_id, str(cls_id))
            if cls_name == "traffic-dataset":
                continue
            detections.append({
                "bbox":       box.xyxy[0].tolist(),
                "class":      cls_name,
                "confidence": round(box.conf[0].item(), 4),
            })
    return detections


def _detect_rfdetr(image: Image.Image, conf: float = 0.5) -> list[dict]:
    dets = state["model"].predict(image, threshold=conf)
    detections = []
    for i in range(len(dets.xyxy)):
        cls_id = int(dets.class_id[i])
        detections.append({
            "bbox":       dets.xyxy[i].tolist(),
            "class":      CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id),
            "confidence": round(float(dets.confidence[i]), 4),
        })
    return detections


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    vram_mb = None
    if torch.cuda.is_available():
        vram_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)
    return {
        "status":        "healthy",
        "modelo_activo": state["model_name"],
        "tipo":          state["model_type"],
        "vram_usada_mb": vram_mb,
    }


@app.get("/models")
async def list_models():
    available = {name: os.path.exists(path) for name, path in MODEL_REGISTRY.items()}
    return {
        "modelo_activo": state["model_name"],
        "modelos":       available,
    }


@app.post("/models/load/{model_name}")
async def load_model(model_name: str):
    with _lock:
        if state["model_name"] == model_name:
            return {"status": "ya_cargado", "modelo": model_name}
        _load_model(model_name)
    return {"status": "cargado", "modelo": model_name, "tipo": state["model_type"]}


@app.post("/models/unload")
async def unload_model():
    with _lock:
        name = state["model_name"]
        _unload_current()
    return {"status": "descargado", "modelo_anterior": name}


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.5):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo no es una imagen")

    with _lock:
        if state["model"] is None:
            raise HTTPException(
                status_code=400,
                detail="Ningún modelo cargado. Usa POST /models/load/{model_name} primero",
            )
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        if state["model_type"] == "rfdetr":
            detections = _detect_rfdetr(image, conf=conf)
        else:
            detections = _detect_ultralytics(image)

    return JSONResponse(content={
        "modelo":      state["model_name"],
        "detecciones": detections,
        "total":       len(detections),
    })


@app.post("/track")
async def track(file: UploadFile = File(...)):
    """Tracking sobre vídeo — devuelve JSON con detecciones por frame.
    RF-DETR no soporta tracking nativo; usa detección frame a frame."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="El archivo no es un vídeo")

    with _lock:
        if state["model"] is None:
            raise HTTPException(
                status_code=400,
                detail="Ningún modelo cargado. Usa POST /models/load/{model_name} primero",
            )
        if state["model_type"] == "rfdetr":
            raise HTTPException(
                status_code=400,
                detail="RF-DETR no soporta tracking con BotSORT. Usa un modelo Ultralytics para tracking.",
            )

        contents = await file.read()
        suffix = "." + file.filename.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            results = state["model"].track(tmp_path, tracker="botsort.yaml", persist=True)
            frames = []
            for frame_idx, r in enumerate(results):
                frame_tracks = []
                if r.boxes.id is not None:
                    for box, track_id in zip(r.boxes, r.boxes.id):
                        cls_id   = int(box.cls[0].item())
                        cls_name = state["model"].names.get(cls_id, str(cls_id))
                        if cls_name == "traffic-dataset":
                            continue
                        frame_tracks.append({
                            "track_id":   int(track_id.item()),
                            "bbox":       box.xyxy[0].tolist(),
                            "class":      cls_name,
                            "confidence": round(box.conf[0].item(), 4),
                        })
                frames.append({"frame": frame_idx, "tracks": frame_tracks})
        finally:
            os.unlink(tmp_path)

    return JSONResponse(content={
        "modelo":       state["model_name"],
        "total_frames": len(frames),
        "frames":       frames,
    })


@app.post("/video")
async def process_video(
    file: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.45,
):
    """Tracking sobre vídeo — devuelve el vídeo anotado con bboxes y track IDs.
    Solo disponible para modelos Ultralytics."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="El archivo no es un vídeo")

    with _lock:
        if state["model"] is None:
            raise HTTPException(
                status_code=400,
                detail="Ningún modelo cargado. Usa POST /models/load/{model_name} primero",
            )
        if state["model_type"] == "rfdetr":
            raise HTTPException(
                status_code=400,
                detail="RF-DETR no soporta tracking con BotSORT. Usa un modelo Ultralytics para /video.",
            )

        contents = await file.read()
        suffix = "." + file.filename.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            tmp_in.write(contents)
            tmp_in_path = tmp_in.name

        out_dir = tempfile.mkdtemp()

        try:
            state["model"].track(
                source=tmp_in_path,
                tracker="botsort.yaml",
                persist=True,
                conf=conf,
                iou=iou,
                save=True,
                project=out_dir,
                name="output",
                exist_ok=True,
            )

            out_folder  = os.path.join(out_dir, "output")
            video_files = [f for f in os.listdir(out_folder) if f.endswith((".mp4", ".avi", ".mov"))]

            if not video_files:
                raise HTTPException(status_code=500, detail="No se generó vídeo de salida")

            out_video_path = os.path.join(out_folder, video_files[0])

        finally:
            os.unlink(tmp_in_path)

    filename = f"{state['model_name']}_{file.filename}"
    return FileResponse(path=out_video_path, media_type="video/mp4", filename=filename)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
