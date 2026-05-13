"""
Corrige el dataset YOLO:
  1. Elimina la clase espuria 'traffic-dataset' (class_id=0, sin instancias).
  2. Resta 1 a todos los class_id → clases 1-7 pasan a ser 0-6.
  3. Actualiza dataset.yaml.

Hace un backup de los labels originales antes de modificar.
"""

import os
import shutil
from pathlib import Path

DATASET_DIR = Path("dataset_yolo")
LABELS_DIRS = [DATASET_DIR / "labels" / "train", DATASET_DIR / "labels" / "val"]
YAML_PATH   = DATASET_DIR / "dataset.yaml"
BACKUP_DIR  = DATASET_DIR / "labels_backup"

EXPECTED_MAX_CLASS = 7   # original: 0 (spurio) + 1-7 (reales)
NEW_MAX_CLASS      = 6   # tras restar 1: 0-6


def backup_labels():
    if BACKUP_DIR.exists():
        print(f"[INFO] Backup ya existe en {BACKUP_DIR}, se omite.")
        return
    print(f"[INFO] Creando backup en {BACKUP_DIR}...")
    shutil.copytree(DATASET_DIR / "labels", BACKUP_DIR)
    print(f"[OK]   Backup creado.")


def fix_label_file(txt_path: Path) -> tuple[int, int]:
    """
    Devuelve (líneas_ok, líneas_con_clase_0_eliminada).
    Reescribe el archivo con class_id - 1 para todos.
    """
    lines_ok = 0
    lines_dropped = 0
    new_lines = []

    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue

            if cid == 0:
                # Era 'traffic-dataset': descartar
                lines_dropped += 1
                continue

            new_cid = cid - 1
            new_lines.append(f"{new_cid} {' '.join(parts[1:])}")
            lines_ok += 1

    with open(txt_path, "w") as f:
        f.write("\n".join(new_lines))
        if new_lines:
            f.write("\n")

    return lines_ok, lines_dropped


def update_yaml():
    new_content = """\
path: dataset_yolo
train: images/train
val:   images/val
names:
  - bicycle
  - bus
  - car
  - human
  - motorcycle
  - trafficcone
  - truck
"""
    with open(YAML_PATH, "w") as f:
        f.write(new_content)
    print(f"[OK]   dataset.yaml actualizado (nc=7, clases 0-6).")


def main():
    print("=" * 60)
    print("  Fix dataset: eliminar clase espuria 'traffic-dataset'")
    print("=" * 60)

    backup_labels()

    total_files     = 0
    total_ok        = 0
    total_dropped   = 0
    total_empty     = 0

    for labels_dir in LABELS_DIRS:
        if not labels_dir.exists():
            print(f"[WARN] No encontrado: {labels_dir}")
            continue

        txt_files = list(labels_dir.glob("*.txt"))
        print(f"\n[INFO] Procesando {labels_dir} ({len(txt_files)} archivos)...")

        for txt_path in txt_files:
            ok, dropped = fix_label_file(txt_path)
            total_files  += 1
            total_ok     += ok
            total_dropped += dropped
            if ok == 0:
                total_empty += 1

    print(f"\n{'='*60}")
    print(f"  Archivos procesados       : {total_files:>8,}")
    print(f"  Anotaciones reindexadas   : {total_ok:>8,}")
    print(f"  Anotaciones class=0 borrad: {total_dropped:>8,}  (eran 'traffic-dataset')")
    print(f"  Archivos vacíos resultantes: {total_empty:>7,}  (sin objetos reales)")
    print(f"{'='*60}")

    update_yaml()

    print(f"\n[OK] Dataset corregido.")
    print(f"     Backup en: {BACKUP_DIR}")
    print(f"\n  Distribución esperada tras corrección:")
    print(f"    0=bicycle  1=bus  2=car  3=human  4=motorcycle  5=trafficcone  6=truck")
    print(f"\n  Recuerda:")
    print(f"    - Borrar labels_backup/ cuando confirmes que todo es correcto.")
    print(f"    - Borrar labels/train.cache y labels/val.cache (Ultralytics los regenera).")


if __name__ == "__main__":
    main()
