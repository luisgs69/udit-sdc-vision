import os
import csv
from pathlib import Path
from ultralytics import YOLO

DATASET_YAML = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'dataset_yolo', 'dataset.yaml')
)

RUNS_BASE = Path('/ultralytics/runs/detect/vision_proyect')

MODELS = {
    'baseline_yolov8n':       'baseline_yolov8n/weights/best.pt',
    'finetuning_yolov8s':     'finetuning_yolov8s_opt/weights/best.pt',
    'frozen_yolov8s':         'frozen_yolov8s/weights/best.pt',
    'augmented_yolov8s':      'augmented_yolov8s/weights/best.pt',
    'finetuning_yolo11s':     'finetuning_yolo11s/weights/best.pt',
}

CLASS_NAMES = ['bicycle', 'bus', 'car', 'human', 'motorcycle', 'trafficcone', 'truck']


def evaluate_model(name, weights_path):
    print(f"\n{'='*60}")
    print(f"Evaluando: {name}")
    print(f"Pesos:     {weights_path}")
    print('='*60)

    if not weights_path.exists():
        print(f"  [SKIP] No encontrado: {weights_path}")
        return None

    model = YOLO(str(weights_path))
    results = model.val(
        data=DATASET_YAML,
        split='val',
        imgsz=640,
        batch=32,
        device=0,
        verbose=False,
    )

    d = results.results_dict
    row = {
        'modelo':      name,
        'mAP50':       round(d.get('metrics/mAP50(B)',    0), 4),
        'mAP50-95':    round(d.get('metrics/mAP50-95(B)', 0), 4),
        'precision':   round(d.get('metrics/precision(B)', 0), 4),
        'recall':      round(d.get('metrics/recall(B)',    0), 4),
    }

    # mAP50 por clase (results.box.ap50 tiene una entrada por clase)
    if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
        for i, cls in enumerate(CLASS_NAMES):
            if i < len(results.box.ap50):
                row[f'AP50_{cls}'] = round(float(results.box.ap50[i]), 4)

    return row


def print_table(rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    col_w = {k: max(len(k), max(len(str(r.get(k, ''))) for r in rows)) for k in keys}
    header = '  '.join(k.ljust(col_w[k]) for k in keys)
    print('\n' + header)
    print('-' * len(header))
    for r in rows:
        print('  '.join(str(r.get(k, '-')).ljust(col_w[k]) for k in keys))


def save_csv(rows, path='resultados_comparativa.csv'):
    if not rows:
        return
    out = Path(__file__).parent / path
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResultados guardados en: {out}")


def main():
    print(f"Dataset: {DATASET_YAML}")
    print(f"Split de evaluación: val (10.571 imágenes)\n")

    rows = []
    for name, rel_path in MODELS.items():
        weights = RUNS_BASE / rel_path
        row = evaluate_model(name, weights)
        if row:
            rows.append(row)

    print(f"\n{'='*60}")
    print("COMPARATIVA FINAL")
    print('='*60)
    print_table(rows)
    save_csv(rows)

    if rows:
        best = max(rows, key=lambda r: r['mAP50-95'])
        print(f"\nMejor modelo (mAP50-95): {best['modelo']}  →  {best['mAP50-95']}")


if __name__ == '__main__':
    main()
