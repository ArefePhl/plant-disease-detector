"""Generate a Grad-CAM heatmap for a single image."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/gradcam.png")
    parser.add_argument("--weights", default="weights/resnet18_best.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    activations = []
    gradients = []
    target_layer = model.layer4[-1].conv2
    target_layer.register_forward_hook(lambda _, __, output: activations.append(output))
    target_layer.register_full_backward_hook(lambda _, grad_in, grad_out: gradients.append(grad_out[0]))

    image = Image.open(args.image).convert("RGB")
    preprocessing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tensor = preprocessing(image).unsqueeze(0).to(device)
    output = model(tensor)
    class_index = output.argmax(dim=1)
    model.zero_grad(set_to_none=True)
    output[0, class_index].backward()

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations[0]).sum(dim=1).relu()[0]
    cam = cam / (cam.max() + 1e-8)
    cam = torch.nn.functional.interpolate(
        cam[None, None], size=(224, 224), mode="bilinear", align_corners=False
    )[0, 0].detach().cpu().numpy()

    original = np.asarray(image.resize((224, 224))) / 255.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(original)
    axis.imshow(cam, cmap="jet", alpha=0.45)
    axis.set_title(f"Grad-CAM: {class_names[class_index.item()]}")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved Grad-CAM to {output_path}")


if __name__ == "__main__":
    main()
