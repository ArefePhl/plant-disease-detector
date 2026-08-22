"""Analyze model errors and the most common confusion pairs on test data."""

import csv
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


DATA_DIR = Path("data/split/test")
MODEL_PATH = Path("weights/resnet18_best.pth")
OUTPUT_DIR = Path("outputs")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    class_names = checkpoint.get("class_names")
    dataset = datasets.ImageFolder(
        DATA_DIR,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
    )
    if not class_names:
        class_names = dataset.classes
    if dataset.classes != class_names:
        raise ValueError("Checkpoint and test dataset classes do not match.")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    errors = []
    confusion_pairs = Counter()
    sample_offset = 0
    with torch.no_grad():
        for inputs, labels in loader:
            probabilities = torch.softmax(model(inputs.to(device)), dim=1)
            confidence, predictions = probabilities.max(dim=1)
            for index, label, prediction, score in zip(range(len(labels)), labels, predictions, confidence):
                if label.item() != prediction.item():
                    path, _ = dataset.samples[sample_offset + index]
                    errors.append({
                        "image": str(path),
                        "actual": class_names[label.item()],
                        "predicted": class_names[prediction.item()],
                        "confidence": round(score.item(), 6),
                    })
                    confusion_pairs[
                        (class_names[label.item()], class_names[prediction.item()])
                    ] += 1
            sample_offset += len(labels)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "misclassified_images.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image", "actual", "predicted", "confidence"]
        )
        writer.writeheader()
        writer.writerows(errors)

    print(f"Misclassified images: {len(errors)}")
    print(f"Saved: {csv_path}")
    print("Most common confusion pairs:")
    for (actual, predicted), count in confusion_pairs.most_common(15):
        print(f"{count:4d} | {actual} -> {predicted}")


if __name__ == "__main__":
    main()
