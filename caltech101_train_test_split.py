import os
import shutil
import random
import argparse
from pathlib import Path


def split_dataset(
    dataset_dir: str,
    output_dir: str,
    test_ratio: float = 0.2,
    seed: int = 42,
    balanced: bool = False,
):
    """
    Split a categorized image dataset into train and test subsets.

    Args:
        dataset_dir: Path to the original dataset directory.
        output_dir:  Path where 'train' and 'test' folders will be created.
        test_ratio:  Fraction of images per category assigned to the test set.
        seed:        Random seed for reproducibility.
        balanced:    If True, ensure all classes have the same number of images.
                     Uses the class with the minimum images as the limit.
    """
    random.seed(seed)

    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)

    train_root = output_path / "train"
    test_root = output_path / "test"

    categories = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    if not categories:
        raise ValueError(f"No subdirectories found in '{dataset_dir}'.")

    print(f"Found {len(categories)} categories.\n")

    # First pass: collect image counts for each category
    category_counts = {}
    for category in categories:
        images = sorted([
            f for f in category.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
        ])
        category_counts[category] = len(images)

    # Determine the minimum count for balancing
    min_count = min(category_counts.values()) if category_counts else 0
    if balanced and min_count > 0:
        print(f"Balanced mode: Using minimum class size of {min_count} images per category.\n")

    total_train = total_test = 0

    for category in categories:
        images = sorted([
            f for f in category.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
        ])

        if not images:
            print(f"  [{category.name}] No images found – skipping.")
            continue

        # Apply balancing if enabled
        if balanced and len(images) > min_count:
            images = images[:min_count]

        random.shuffle(images)
        n_test = max(1, round(len(images) * test_ratio))
        test_images = images[:n_test]
        train_images = images[n_test:]

        for split_root, split_images in [(train_root, train_images), (test_root, test_images)]:
            dest_dir = split_root / category.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                shutil.copy2(img, dest_dir / img.name)

        print(f"  [{category.name}]  total={len(images)}  train={len(train_images)}  test={len(test_images)}")
        total_train += len(train_images)
        total_test += len(test_images)

    print(f"\nDone!  train={total_train} images  |  test={total_test} images")
    print(f"Output written to: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a categorized image dataset into train/test sets.")
    parser.add_argument("dataset_dir", help="Path to the original dataset directory.")
    parser.add_argument("output_dir", help="Directory where 'train' and 'test' folders will be created.")
    parser.add_argument(
        "--test_ratio", type=float, default=0.2,
        help="Fraction of images per category used for testing (default: 0.2)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--balanced", action="store_true",
        help="If set, ensure all classes have the same number of images. Uses the class with the minimum images as the limit."
    )
    args = parser.parse_args()

    split_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
        balanced=args.balanced,
    )