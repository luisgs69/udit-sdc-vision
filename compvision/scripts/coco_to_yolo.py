import json
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import random
import numpy as np
from collections import Counter
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import yaml

def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    return (
        (x + w / 2) / img_w,
        (y + h / 2) / img_h,
        w / img_w,
        h / img_h,
    )


def stratified_split_by_instances_iterstrat(
    images,
    anns_by_image,
    cat_id_to_yolo_id,
    num_classes,
    val_size=0.2,
    seed=42,
):
    image_ids = list(images.keys())

    # Matriz Y: filas = imágenes, columnas = clases
    # Valor = número de instancias de esa clase en esa imagen
    y = np.zeros((len(image_ids), num_classes), dtype=int)

    for i, image_id in enumerate(image_ids):
        for ann in anns_by_image[image_id]:
            cls = cat_id_to_yolo_id[ann["category_id"]]
            y[i, cls] += 1

    X = np.zeros((len(image_ids), 1))

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=seed,
    )

    train_idx, val_idx = next(splitter.split(X, y))

    train_ids = [image_ids[i] for i in train_idx]
    val_ids = [image_ids[i] for i in val_idx]

    return train_ids, val_ids

def main():
    coco_json = Path(r"C:\Users\Luis\Desktop\Master\Vision\trabajo\train\labels.json")
    images_dir = Path(r"C:\Users\Luis\Desktop\Master\Vision\trabajo\train\data")
    output_dir = Path(r"C:\Users\Luis\Desktop\Master\Vision\trabajo\dataset_yolo")

    val_size = 0.2
    seed = 42

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_yolo_id = {cat["id"]: i for i, cat in enumerate(categories)}
    class_names = [cat["name"] for cat in categories]

    images = {img["id"]: img for img in coco["images"]}

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    train_ids, val_ids = stratified_split_by_instances_iterstrat(
        images=images,
        anns_by_image=anns_by_image,
        cat_id_to_yolo_id=cat_id_to_yolo_id,
        num_classes=len(class_names),
        val_size=val_size,
        seed=seed,
)

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        for image_id in ids:
            img = images[image_id]

            src_img = images_dir / img["file_name"]
            dst_img = output_dir / "images" / split / img["file_name"]

            shutil.copy2(src_img, dst_img)

            label_path = output_dir / "labels" / split / f"{Path(img['file_name']).stem}.txt"

            lines = []
            for ann in anns_by_image[image_id]:
                class_id = cat_id_to_yolo_id[ann["category_id"]]

                x, y, w, h = coco_bbox_to_yolo(
                    ann["bbox"],
                    img["width"],
                    img["height"],
                )

                lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            label_path.write_text("\n".join(lines), encoding="utf-8")

    dataset_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
    }

    with open(output_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False, allow_unicode=True)

    print("Dataset YOLO creado en:", output_dir)
    print()
    print("Resumen de estratificación por instancias:")
    print(f"Imágenes train: {len(train_ids)}")
    print(f"Imágenes val:   {len(val_ids)}")
    print()


    # Calculate distribution
    train_counts = Counter()
    val_counts = Counter()
    
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for image_id in ids:
            for ann in anns_by_image[image_id]:
                cls_id = cat_id_to_yolo_id[ann["category_id"]]
                if split == "train":
                    train_counts[cls_id] += 1
                else:
                    val_counts[cls_id] += 1

    total_counts = train_counts + val_counts

    print("\nResumen de distribución por clase:")
    for cls_id, name in enumerate(class_names):
        total = total_counts[cls_id]
        val = val_counts[cls_id]
        train = train_counts[cls_id]
        pct_val = val / total * 100 if total else 0

        print(
            f"{cls_id} - {name}: "
            f"total={total}, train={train}, val={val}, val={pct_val:.2f}%"
        )


if __name__ == "__main__":
    main()