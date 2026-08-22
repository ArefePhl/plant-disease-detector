import os
import shutil
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join("data", "raw")
OUTPUT_DIR = os.path.join("data", "split")
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_dataset():
    if not os.path.isdir(RAW_DIR):
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DIR}")

    # پاک‌سازی کامل split قبلی تا فایل‌های قدیمی یا تکراری باقی نمانند.
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    total_counts = {name: 0 for name in RATIOS}
    class_names = sorted(
        name for name in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, name))
    )

    for class_name in class_names:
        class_path = os.path.join(RAW_DIR, class_name)
        images = sorted(
            name for name in os.listdir(class_path)
            if os.path.splitext(name)[1].lower() in VALID_EXTENSIONS
        )
        if len(images) < 3:
            raise ValueError(f"Class '{class_name}' must contain at least 3 images.")

        train_imgs, remainder = train_test_split(
            images, train_size=RATIOS["train"], random_state=SEED
        )
        val_fraction = RATIOS["val"] / (RATIOS["val"] + RATIOS["test"])
        val_imgs, test_imgs = train_test_split(
            remainder, train_size=val_fraction, random_state=SEED
        )

        for split_name, split_imgs in (
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ):
            split_class_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for image_name in split_imgs:
                shutil.copy2(
                    os.path.join(class_path, image_name),
                    os.path.join(split_class_dir, image_name),
                )
            total_counts[split_name] += len(split_imgs)

    print(
        "Data splitting completed: "
        f"train={total_counts['train']}, "
        f"val={total_counts['val']}, "
        f"test={total_counts['test']}"
    )


if __name__ == "__main__":
    split_dataset()
