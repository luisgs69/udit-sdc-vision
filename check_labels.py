"""
Verificación de calidad de etiquetas YOLO.
Genera:
  - Informe estadístico por consola
  - label_issues.txt  con las anotaciones sospechosas
  - grid_samples.jpg  con N imágenes aleatorias anotadas
  - grid_suspicious.jpg con las M anotaciones más raras
"""

import os
import sys
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─── Configuración ────────────────────────────────────────────────────────────
CLASSES = ["bicycle", "bus", "car", "human", "motorcycle", "trafficcone", "truck"]
COLORS  = [
    (255, 80, 80),    # bicycle   - rojo
    (80, 200, 80),    # bus       - verde
    (80, 80, 255),    # car       - azul
    (255, 200, 0),    # human     - amarillo
    (200, 80, 255),   # motorcycle- violeta
    (0, 220, 220),    # trafficcone- cyan
    (255, 140, 0),    # truck     - naranja
]

# Umbrales para detección de anomalías
MIN_AREA  = 0.0001   # < 0.01% del área de la imagen → bbox microscópica
MAX_AREA  = 0.80     # > 80% del área                → bbox gigante
MIN_DIM   = 0.004    # ancho o alto < 0.4%            → dimensión casi cero
MAX_COORD = 1.05     # tolerancia mínima fuera de [0,1]


def parse_label(txt_path):
    """Devuelve lista de (class_id, xc, yc, w, h) o [] si vacío."""
    rows = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                rows.append(None)  # línea malformada
                continue
            try:
                cid, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                rows.append((cid, xc, yc, w, h))
            except ValueError:
                rows.append(None)
    return rows


def classify_issue(cid, xc, yc, w, h):
    """Devuelve lista de strings con los problemas encontrados."""
    issues = []
    if cid < 0 or cid >= len(CLASSES):
        issues.append(f"class_id={cid} fuera de rango [0,{len(CLASSES)-1}]")
    coords = {"xc": xc, "yc": yc, "w": w, "h": h}
    for name, val in coords.items():
        if val < 0 or val > MAX_COORD:
            issues.append(f"{name}={val:.4f} fuera de [0,1]")
    if w <= 0 or h <= 0:
        issues.append(f"dimensión no positiva w={w:.4f} h={h:.4f}")
    if w < MIN_DIM:
        issues.append(f"ancho microscópico w={w:.4f}")
    if h < MIN_DIM:
        issues.append(f"alto microscópico h={h:.4f}")
    area = w * h
    if area < MIN_AREA:
        issues.append(f"área microscópica={area:.6f}")
    if area > MAX_AREA:
        issues.append(f"área gigante={area:.3f}")
    return issues


def draw_boxes(img_path, labels, size=640):
    """Devuelve imagen PIL redimensionada con bboxes dibujadas."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (size, size), (50, 50, 50))
    W, H = img.size
    img = img.resize((size, size))
    draw = ImageDraw.Draw(img)
    sx, sy = size / W, size / H
    for ann in labels:
        if ann is None:
            continue
        cid, xc, yc, w, h = ann
        x1 = int((xc - w / 2) * W * sx)
        y1 = int((yc - h / 2) * H * sy)
        x2 = int((xc + w / 2) * W * sx)
        y2 = int((yc + h / 2) * H * sy)
        color = COLORS[cid % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = CLASSES[cid] if cid < len(CLASSES) else str(cid)
        draw.text((x1 + 2, y1 + 2), label, fill=color)
    return img


def make_grid(images, cols=4, cell=320, title=""):
    """Ensambla una cuadrícula de imágenes PIL."""
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell, rows * cell + 30), (30, 30, 30))
    draw = ImageDraw.Draw(grid)
    draw.text((4, 4), title, fill=(220, 220, 220))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        thumb = img.resize((cell, cell))
        grid.paste(thumb, (c * cell, r * cell + 30))
    return grid


def scan_split(images_dir, labels_dir):
    """Escanea todos los archivos de un split y devuelve estadísticas."""
    label_files = sorted(Path(labels_dir).glob("*.txt"))
    img_dir = Path(images_dir)

    stats = {
        "total_files": len(label_files),
        "empty_files": 0,
        "malformed_lines": 0,
        "total_annotations": 0,
        "issues": [],           # (img_name, ann, issue_list)
        "class_counts": Counter(),
        "areas": [],
        "widths": [],
        "heights": [],
        "missing_image": 0,
    }

    for txt_path in label_files:
        stem = txt_path.stem
        # Buscar imagen correspondiente
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            p = img_dir / (stem + ext)
            if p.exists():
                img_path = p
                break
        if img_path is None:
            stats["missing_image"] += 1

        labels = parse_label(txt_path)

        if not labels:
            stats["empty_files"] += 1
            continue

        for ann in labels:
            if ann is None:
                stats["malformed_lines"] += 1
                continue
            cid, xc, yc, w, h = ann
            stats["total_annotations"] += 1
            stats["class_counts"][cid] += 1
            stats["areas"].append(w * h)
            stats["widths"].append(w)
            stats["heights"].append(h)

            issues = classify_issue(cid, xc, yc, w, h)
            if issues:
                stats["issues"].append((txt_path.name, (img_path, ann), issues))

    return stats


def print_report(split_name, stats):
    print(f"\n{'='*60}")
    print(f"  Split: {split_name.upper()}")
    print(f"{'='*60}")
    print(f"  Archivos de etiqueta      : {stats['total_files']:>8,}")
    print(f"  Imágenes sin archivo label: {stats['missing_image']:>8,}")
    print(f"  Labels vacíos (0 objetos) : {stats['empty_files']:>8,}")
    print(f"  Líneas malformadas        : {stats['malformed_lines']:>8,}")
    print(f"  Total anotaciones         : {stats['total_annotations']:>8,}")
    print(f"  Anotaciones con problemas : {len(stats['issues']):>8,}")

    print(f"\n  Distribución de clases:")
    total = stats["total_annotations"] or 1
    for cid, name in enumerate(CLASSES):
        cnt = stats["class_counts"].get(cid, 0)
        bar = "█" * int(30 * cnt / total)
        print(f"    [{cid}] {name:<14} {cnt:>8,}  {cnt/total*100:5.1f}%  {bar}")

    if stats["areas"]:
        areas = np.array(stats["areas"])
        ws    = np.array(stats["widths"])
        hs    = np.array(stats["heights"])
        print(f"\n  Estadísticas de bounding boxes (relativas a imagen):")
        print(f"    Área  — media={areas.mean():.4f}  mediana={np.median(areas):.4f}  "
              f"p1={np.percentile(areas,1):.5f}  p99={np.percentile(areas,99):.4f}")
        print(f"    Ancho — media={ws.mean():.4f}  mediana={np.median(ws):.4f}  "
              f"p1={np.percentile(ws,1):.5f}  p99={np.percentile(ws,99):.4f}")
        print(f"    Alto  — media={hs.mean():.4f}  mediana={np.median(hs):.4f}  "
              f"p1={np.percentile(hs,1):.5f}  p99={np.percentile(hs,99):.4f}")

    if stats["issues"]:
        print(f"\n  ⚠  Primeras 20 anotaciones problemáticas:")
        for name, (_, ann), issues in stats["issues"][:20]:
            print(f"    {name}: {issues}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_yolo",
                        help="Ruta al directorio del dataset YOLO")
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "both"])
    parser.add_argument("--samples", type=int, default=32,
                        help="Número de imágenes aleatorias en el grid")
    parser.add_argument("--suspicious", type=int, default=16,
                        help="Número de imágenes sospechosas en el grid")
    parser.add_argument("--out", default=".",
                        help="Directorio de salida para los grids")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    splits  = ["train", "val"] if args.split == "both" else [args.split]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_issues = []

    for split in splits:
        images_dir = dataset / "images" / split
        labels_dir = dataset / "labels" / split

        if not labels_dir.exists():
            print(f"[WARN] No encontrado: {labels_dir}")
            continue

        print(f"\nEscaneando {split}... ({len(list(labels_dir.glob('*.txt')))} archivos)")
        stats = scan_split(images_dir, labels_dir)
        print_report(split, stats)

        # Guardar issues en texto
        issues_path = out_dir / f"label_issues_{split}.txt"
        with open(issues_path, "w") as f:
            for name, _, issues in stats["issues"]:
                f.write(f"{name}: {'; '.join(issues)}\n")
        print(f"\n  Issues guardados en: {issues_path}")

        # Grid de muestras aleatorias
        label_files = list((dataset / "labels" / split).glob("*.txt"))
        sample_files = random.sample(label_files, min(args.samples, len(label_files)))
        imgs_random = []
        for txt_path in sample_files:
            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                p = images_dir / (txt_path.stem + ext)
                if p.exists():
                    img_path = p
                    break
            if img_path:
                labels = [a for a in parse_label(txt_path) if a is not None]
                imgs_random.append(draw_boxes(img_path, labels, size=320))

        if imgs_random:
            grid = make_grid(imgs_random, cols=8, cell=320,
                             title=f"{split} — {args.samples} muestras aleatorias")
            out_path = out_dir / f"grid_samples_{split}.jpg"
            grid.save(out_path, quality=85)
            print(f"  Grid aleatorio guardado en: {out_path}")

        # Grid de anotaciones sospechosas
        imgs_susp = []
        for name, (img_path, ann), issues in stats["issues"][:args.suspicious]:
            if img_path and img_path.exists():
                imgs_susp.append(draw_boxes(img_path, [ann], size=320))

        if imgs_susp:
            grid_s = make_grid(imgs_susp, cols=4, cell=320,
                               title=f"{split} — primeras {len(imgs_susp)} anotaciones sospechosas")
            out_path_s = out_dir / f"grid_suspicious_{split}.jpg"
            grid_s.save(out_path_s, quality=85)
            print(f"  Grid sospechoso guardado en: {out_path_s}")

        all_issues.extend(stats["issues"])

    total_issues = len(all_issues)
    print(f"\n{'='*60}")
    print(f"  RESUMEN GLOBAL: {total_issues} anotaciones con posibles errores")
    if total_issues == 0:
        print("  ✓ No se detectaron anomalías automáticas.")
    else:
        print("  Revisar los grids generados para validación visual.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
