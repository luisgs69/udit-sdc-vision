"""
Evaluación con métricas COCO para cualquier modelo entrenado.
Métricas según metricas.pdf: IoU, TP, FP, FN, Precision, Recall, AP, mAP.

Uso:
  # Modelo Ultralytics (.pt)
  python3 evaluate_coco_metrics.py --model yolov8s --weights runs/detect/.../best.pt

  # RF-DETR (.pth)
  python3 evaluate_coco_metrics.py --model rfdetr --weights runs/.../checkpoint_best_regular.pth
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASSES = ["bicycle", "bus", "car", "human", "motorcycle", "trafficcone", "truck"]
VAL_COCO_JSON = "dataset_coco/valid/_annotations.coco.json"
VAL_IMAGES    = "dataset_coco/valid"


def compute_iou(boxA, boxB):
    """IoU entre dos cajas [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (aA + aB - inter + 1e-9)


# ─── Ultralytics ──────────────────────────────────────────────────────────────

def evaluate_ultralytics(weights: str, data_yaml: str = "dataset_yolo/dataset.yaml"):
    from ultralytics import YOLO
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, split="val", verbose=False)

    print("\n" + "="*65)
    print(f"  MÉTRICAS COCO — {Path(weights).stem}")
    print("="*65)
    print(f"  mAP@50      : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95   : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")
    print()
    print(f"  {'Clase':<14} {'AP@50':>8} {'AP@50-95':>10}")
    print(f"  {'-'*36}")
    for i, name in enumerate(model.names.values()):
        if name == "traffic-dataset":
            continue
        ap50   = metrics.box.ap50[i]   if i < len(metrics.box.ap50)   else 0
        ap5095 = metrics.box.maps[i]   if i < len(metrics.box.maps)   else 0
        print(f"  {name:<14} {ap50:>8.4f} {ap5095:>10.4f}")
    print("="*65)
    return metrics


# ─── RF-DETR con pycocotools ──────────────────────────────────────────────────

def evaluate_rfdetr(weights: str, conf_thresh: float = 0.01):
    from rfdetr import RFDETRBase

    print(f"Cargando RF-DETR desde {weights}...")
    model = RFDETRBase(pretrain_weights=weights)

    coco_gt = COCO(VAL_COCO_JSON)
    img_ids  = coco_gt.getImgIds()
    results  = []

    print(f"Evaluando {len(img_ids)} imágenes en validación...")
    for i, img_id in enumerate(img_ids):
        if i % 1000 == 0:
            print(f"  {i}/{len(img_ids)}")

        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(VAL_IMAGES, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        pil_img = Image.open(img_path).convert("RGB")
        dets = model.predict(pil_img, threshold=conf_thresh)

        if len(dets.xyxy) == 0:
            continue

        for j in range(len(dets.xyxy)):
            x1, y1, x2, y2 = dets.xyxy[j].tolist()
            results.append({
                "image_id":    img_id,
                "category_id": int(dets.class_id[j]),
                "bbox":        [x1, y1, x2 - x1, y2 - y1],  # COCO: [x,y,w,h]
                "score":       float(dets.confidence[j]),
            })

    if not results:
        print("Sin predicciones — revisa el modelo o el threshold.")
        return

    tmp_results = "/tmp/rfdetr_results.json"
    with open(tmp_results, "w") as f:
        json.dump(results, f)

    coco_dt  = coco_gt.loadRes(tmp_results)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()

    print("\n" + "="*65)
    print(f"  MÉTRICAS COCO — {Path(weights).stem}")
    print("="*65)
    evaluator.summarize()

    # Por clase
    cat_ids = coco_gt.getCatIds()
    cats    = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
    print(f"\n  {'Clase':<14} {'AP@50':>8} {'AP@50-95':>10}")
    print(f"  {'-'*36}")
    for cat_id in cat_ids:
        evaluator.params.catIds = [cat_id]
        evaluator.evaluate()
        evaluator.accumulate()
        stats = evaluator.stats
        print(f"  {cats[cat_id]:<14} {stats[1]:>8.4f} {stats[0]:>10.4f}")

    print("="*65)
    evaluator.params.catIds = cat_ids  # restaurar
    return evaluator


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, choices=["ultralytics", "rfdetr"],
                        help="Tipo de modelo")
    parser.add_argument("--weights", required=True, help="Ruta al archivo de pesos (.pt o .pth)")
    parser.add_argument("--data",    default="dataset_yolo/dataset.yaml",
                        help="dataset.yaml para modelos Ultralytics")
    parser.add_argument("--conf",    type=float, default=0.01,
                        help="Confidence threshold para RF-DETR (bajo para mAP completo)")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"[ERROR] Weights no encontrados: {args.weights}")
        return

    if args.model == "ultralytics":
        evaluate_ultralytics(args.weights, args.data)
    else:
        evaluate_rfdetr(args.weights, args.conf)


if __name__ == "__main__":
    main()
