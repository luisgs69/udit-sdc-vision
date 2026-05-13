"""
clean_classes.py — Elimina clases vacías y renumera en COCO + YOLO.

Uso por defecto (detecta y elimina automáticamente clases con 0 anotaciones):
    python3 clean_classes.py

Especificando rutas distintas:
    python3 clean_classes.py \
        --coco_dirs dataset_coco/train dataset_coco/valid \
        --yolo_label_dirs dataset_yolo/labels/train dataset_yolo/labels/val \
        --dataset_yaml dataset_yolo/dataset.yaml \
        --no_backup
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


def load_coco(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_coco(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def find_empty_categories(coco: dict) -> set:
    used = {ann["category_id"] for ann in coco.get("annotations", [])}
    return {cat["id"] for cat in coco["categories"] if cat["id"] not in used}


def build_id_remap(categories: list, remove_ids: set) -> dict:
    """Returns {old_id: new_id} for kept categories, sorted by original id."""
    kept = sorted(cat for cat in categories if cat["id"] not in remove_ids, key=lambda c: c["id"])
    return {cat["id"]: new_id for new_id, cat in enumerate(kept)}


def patch_coco(coco: dict, remap: dict, remove_ids: set) -> dict:
    coco["categories"] = [
        {**cat, "id": remap[cat["id"]]}
        for cat in sorted(coco["categories"], key=lambda c: c["id"])
        if cat["id"] not in remove_ids
    ]
    coco["annotations"] = [
        {**ann, "category_id": remap[ann["category_id"]]}
        for ann in coco["annotations"]
        if ann["category_id"] not in remove_ids
    ]
    return coco


def patch_yolo_file(path: Path, remap: dict, remove_ids: set) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = []
    removed = 0
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        old_id = int(parts[0])
        if old_id in remove_ids:
            removed += 1
            continue
        parts[0] = str(remap[old_id])
        kept.append(" ".join(parts))
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed


def patch_yaml(yaml_path: Path, keep_names: list) -> None:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["names"] = keep_names
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser(description="Elimina clases vacías y renumera COCO + YOLO")
    parser.add_argument(
        "--coco_dirs",
        nargs="+",
        default=["dataset_coco/train", "dataset_coco/valid"],
        help="Carpetas COCO (contienen _annotations.coco.json)",
    )
    parser.add_argument(
        "--yolo_label_dirs",
        nargs="+",
        default=["dataset_yolo/labels/train", "dataset_yolo/labels/val"],
        help="Carpetas con etiquetas YOLO (.txt)",
    )
    parser.add_argument(
        "--dataset_yaml",
        default="dataset_yolo/dataset.yaml",
        help="YAML del dataset YOLO",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="No crear copias de seguridad (.bak)",
    )
    args = parser.parse_args()

    # ── 1. Determinar clases a eliminar usando el primer JSON COCO ──────────
    first_coco_path = Path(args.coco_dirs[0]) / "_annotations.coco.json"
    ref_coco = load_coco(first_coco_path)
    remove_ids = find_empty_categories(ref_coco)

    if not remove_ids:
        print("No se encontraron clases vacías. Nada que hacer.")
        return

    remove_names = {cat["name"] for cat in ref_coco["categories"] if cat["id"] in remove_ids}
    remap = build_id_remap(ref_coco["categories"], remove_ids)
    keep_names = [
        cat["name"]
        for cat in sorted(ref_coco["categories"], key=lambda c: c["id"])
        if cat["id"] not in remove_ids
    ]

    print(f"Clases eliminadas ({len(remove_ids)}): {remove_names}")
    print(f"Remapeo de IDs: {remap}")
    print(f"Clases resultantes ({len(keep_names)}): {keep_names}")
    print()

    # ── 2. Parchear JSON COCO ────────────────────────────────────────────────
    for coco_dir in args.coco_dirs:
        json_path = Path(coco_dir) / "_annotations.coco.json"
        if not json_path.exists():
            print(f"[SKIP] {json_path} no existe")
            continue
        if not args.no_backup:
            shutil.copy2(json_path, json_path.with_suffix(".coco.json.bak"))
        coco = load_coco(json_path)
        coco = patch_coco(coco, remap, remove_ids)
        save_coco(coco, json_path)
        print(f"[COCO] {json_path} → {len(coco['categories'])} clases, {len(coco['annotations'])} anotaciones")

    # ── 3. Parchear etiquetas YOLO ───────────────────────────────────────────
    total_files = 0
    total_removed_lines = 0
    for label_dir in args.yolo_label_dirs:
        label_path = Path(label_dir)
        if not label_path.exists():
            print(f"[SKIP] {label_path} no existe")
            continue
        txt_files = sorted(label_path.glob("*.txt"))
        removed_lines = 0
        for txt in txt_files:
            removed_lines += patch_yolo_file(txt, remap, remove_ids)
        total_files += len(txt_files)
        total_removed_lines += removed_lines
        print(f"[YOLO] {label_path}: {len(txt_files)} archivos, {removed_lines} líneas eliminadas")

    # ── 4. Parchear dataset.yaml ─────────────────────────────────────────────
    yaml_path = Path(args.dataset_yaml)
    if yaml_path.exists():
        if not args.no_backup:
            shutil.copy2(yaml_path, yaml_path.with_suffix(".yaml.bak"))
        patch_yaml(yaml_path, keep_names)
        print(f"[YAML] {yaml_path} actualizado con {len(keep_names)} clases")
    else:
        print(f"[SKIP] {yaml_path} no existe")

    print()
    print("Resumen:")
    print(f"  Clases eliminadas : {remove_names}")
    print(f"  Clases resultantes: {len(keep_names)} → {keep_names}")
    print(f"  Archivos YOLO     : {total_files}")
    print(f"  Líneas YOLO borr. : {total_removed_lines}")
    if not args.no_backup:
        print("  Backups .bak creados para JSON y YAML originales.")


if __name__ == "__main__":
    main()
