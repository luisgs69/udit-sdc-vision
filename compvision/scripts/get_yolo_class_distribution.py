from collections import Counter
from json.tool import main
from pathlib import Path


def get_yolo_class_distribution(dataset_dir):

    dataset_dir = Path(dataset_dir)

    def count_split(split):
        labels_dir = dataset_dir / "labels" / split
        counts = Counter()

        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, "r") as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        counts[class_id] += 1

        return counts

    train_counts = count_split("train")
    val_counts = count_split("val")

    total_counts = train_counts + val_counts

    print("\n📊 Distribución por clase (instancias y estratificación):\n")
    print(f"{'Clase':<10} | {'Total':<8} | {'Train':<8} | {'Val':<8} | {'Train %':<10} | {'Val %':<10}")
    print("-" * 65)

    for cls in sorted(total_counts.keys()):
        total = total_counts[cls]
        train = train_counts[cls]
        val = val_counts[cls]

        pct_train = (train / total * 100) if total else 0
        pct_val = (val / total * 100) if total else 0

        print(
            f"{cls:<10} | {total:<8} | {train:<8} | {val:<8} | {pct_train:<10.2f} | {pct_val:<10.2f}"
        )

    return train_counts, val_counts, total_counts

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Get class distribution from YOLO dataset")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the root folder of the YOLO dataset",
    )
    args = parser.parse_args()

    get_yolo_class_distribution(args.dataset_dir)
