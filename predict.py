"""Predict the plant-disease class for one image."""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a class for one image.")
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument("--weights", default="weights/resnet18_best.pth")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    class_names = checkpoint.get("class_names")
    if not class_names:
        class_names = sorted(
            path.name for path in (Path("data/split") / "train").iterdir() if path.is_dir()
        )
    if not class_names:
        raise ValueError("Checkpoint does not contain class_names.")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(device).eval()

    preprocessing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    if not os.path.isfile(args.image):
        raise FileNotFoundError(args.image)
    image = Image.open(args.image).convert("RGB")
    tensor = preprocessing(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    values, indices = probabilities.topk(min(args.top_k, len(class_names)))
    print(f"Device: {device}")
    print("Predictions:")
    for rank, (value, index) in enumerate(zip(values, indices), start=1):
        print(f"{rank}. {class_names[index.item()]} ({value.item() * 100:.2f}%)")


if __name__ == "__main__":
    main()
