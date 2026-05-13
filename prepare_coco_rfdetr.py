"""
Prepara el dataset COCO para RF-DETR:
  1. Divide labels.json en train.json / val.json usando el mismo split YOLO
  2. Elimina la categoría espuria 'traffic-dataset' (id=0)
  3. Reindexar category_id: 1-7 → 0-6
  4. Muestra informe de estratificación equivalente a get_yolo_class_distribution.py
  5. Genera dataset_coco/ con la estructura esperada por RF-DETR
"""

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# ─── Rutas ────────────────────────────────────────────────────────────────────
COCO_JSON    = Path("train/labels.json")
YOLO_TRAIN   = Path("dataset_yolo/images/train")
YOLO_VAL     = Path("dataset_yolo/images/val")
IMAGES_DIR   = Path("train/data")          # imágenes originales
OUT_DIR      = Path("dataset_coco")

CLASSES = ["bicycle", "bus", "car", "human", "motorcycle", "trafficcone", "truck"]

# ─── Cargar COCO ──────────────────────────────────────────────────────────────
print("Cargando labels.json...")
with open(COCO_JSON) as f:
    coco = json.load(f)

# Índice image_id → imagen
img_by_id   = {img["id"]: img for img in coco["images"]}
# Índice file_name → image_id
img_by_name = {img["file_name"]: img["id"] for img in coco["images"]}

# ─── Leer splits YOLO ─────────────────────────────────────────────────────────
train_names = {p.name for p in YOLO_TRAIN.iterdir() if p.is_file()}
val_names   = {p.name for p in YOLO_VAL.iterdir()   if p.is_file()}

print(f"  Imágenes YOLO train : {len(train_names):,}")
print(f"  Imágenes YOLO val   : {len(val_names):,}")

# ─── Separar imágenes ─────────────────────────────────────────────────────────
train_ids, val_ids = set(), set()
missing = 0
for name, img_id in img_by_name.items():
    if name in train_names:
        train_ids.add(img_id)
    elif name in val_names:
        val_ids.add(img_id)
    else:
        missing += 1

print(f"  Sin match en split  : {missing}")
print(f"  IDs asignados train : {len(train_ids):,}")
print(f"  IDs asignados val   : {len(val_ids):,}")

# ─── Filtrar y reindexar anotaciones ──────────────────────────────────────────
# category_id 0 (traffic-dataset) → eliminar
# category_id 1-7 → 0-6
def remap_cat(cid):
    return cid - 1   # 1→0, 2→1, ..., 7→6

train_anns, val_anns = [], []
for ann in coco["annotations"]:
    if ann["category_id"] == 0:
        continue          # clase espuria
    new_ann = dict(ann, category_id=remap_cat(ann["category_id"]))
    if ann["image_id"] in train_ids:
        train_anns.append(new_ann)
    elif ann["image_id"] in val_ids:
        val_anns.append(new_ann)

# ─── Nuevas categorías ────────────────────────────────────────────────────────
new_categories = [
    {"id": i, "name": name, "supercategory": "traffic"}
    for i, name in enumerate(CLASSES)
]

# ─── Construir JSONs ──────────────────────────────────────────────────────────
def build_coco(image_ids, annotations):
    images = [img_by_id[iid] for iid in sorted(image_ids)]
    return {
        "info":        coco.get("info", {}),
        "licenses":    coco.get("licenses", []),
        "categories":  new_categories,
        "images":      images,
        "annotations": annotations,
    }

train_coco = build_coco(train_ids, train_anns)
val_coco   = build_coco(val_ids,   val_anns)

# ─── Guardar ──────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "train").mkdir(exist_ok=True)
(OUT_DIR / "valid").mkdir(exist_ok=True)

train_json_path = OUT_DIR / "train" / "_annotations.coco.json"
val_json_path   = OUT_DIR / "valid" / "_annotations.coco.json"

print("\nGuardando JSONs...")
with open(train_json_path, "w") as f:
    json.dump(train_coco, f)
print(f"  {train_json_path}")

with open(val_json_path, "w") as f:
    json.dump(val_coco, f)
print(f"  {val_json_path}")

# ─── Estratificación ──────────────────────────────────────────────────────────
train_dist = Counter(a["category_id"] for a in train_anns)
val_dist   = Counter(a["category_id"] for a in val_anns)
total_dist = train_dist + val_dist

print(f"\n{'='*70}")
print(f"  Estratificación por clase (instancias)")
print(f"{'='*70}")
print(f"{'Clase':<14} | {'Total':>8} | {'Train':>8} | {'Val':>8} | {'Train%':>8} | {'Val%':>8}")
print(f"{'-'*70}")

for cid, name in enumerate(CLASSES):
    total = total_dist[cid]
    train = train_dist[cid]
    val   = val_dist[cid]
    pct_t = train / total * 100 if total else 0
    pct_v = val   / total * 100 if total else 0
    print(f"{name:<14} | {total:>8,} | {train:>8,} | {val:>8,} | {pct_t:>7.2f}% | {pct_v:>7.2f}%")

print(f"{'-'*70}")
total_all = sum(total_dist.values())
train_all = sum(train_dist.values())
val_all   = sum(val_dist.values())
print(f"{'TOTAL':<14} | {total_all:>8,} | {train_all:>8,} | {val_all:>8,} | "
      f"{train_all/total_all*100:>7.2f}% | {val_all/total_all*100:>7.2f}%")
print(f"{'='*70}")

print(f"\nResumen:")
print(f"  Imágenes train : {len(train_coco['images']):,}")
print(f"  Imágenes val   : {len(val_coco['images']):,}")
print(f"  Anots. train   : {len(train_anns):,}")
print(f"  Anots. val     : {len(val_anns):,}")
print(f"  Categorías     : {len(CLASSES)} (traffic-dataset eliminado, ids 0-6)")
print(f"\nEstructura generada en {OUT_DIR}/:")
print(f"  train/_annotations.coco.json")
print(f"  valid/_annotations.coco.json")
print(f"\nNota: RF-DETR espera las imágenes en dataset_coco/train/ y dataset_coco/valid/")
print(f"      Las imágenes están en {IMAGES_DIR}/ — crear symlinks o copiarlas.")
