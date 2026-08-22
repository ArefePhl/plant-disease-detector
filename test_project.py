"""Small smoke tests for the project components."""

import torch
from PIL import Image

from train import build_model, make_transforms


def main():
    transforms_by_phase = make_transforms()
    sample = Image.new("RGB", (224, 224))
    normalized = transforms_by_phase["val"](sample)
    assert normalized.shape == (3, 224, 224)
    model = build_model(38, pretrained=False)
    output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 38)
    print("Smoke tests passed.")


if __name__ == "__main__":
    main()
