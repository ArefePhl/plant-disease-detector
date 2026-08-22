"""Train a ResNet-18 plant-disease classifier."""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet-18 classifier.")
    parser.add_argument("--data-dir", default="data/split")
    parser.add_argument("--weights", default="weights/resnet18_best.pth")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms():
    return {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
    }


def build_loaders(data_dir, batch_size, num_workers):
    transforms_by_phase = make_transforms()
    datasets_by_phase = {
        phase: datasets.ImageFolder(
            str(Path(data_dir) / phase), transform=transforms_by_phase[phase]
        )
        for phase in ("train", "val")
    }
    class_names = datasets_by_phase["train"].classes
    if datasets_by_phase["val"].classes != class_names:
        raise ValueError("Train and validation classes do not match.")
    loaders = {
        phase: DataLoader(
            datasets_by_phase[phase],
            batch_size=batch_size,
            shuffle=phase == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for phase in ("train", "val")
    }
    return datasets_by_phase, loaders, class_names


def build_model(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, phase, epoch, epochs):
    training = phase == "train"
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    progress = tqdm(loader, desc=f"{phase.capitalize()} {epoch}/{epochs}", leave=False)

    for inputs, labels in progress:
        inputs, labels = inputs.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_items += batch_size
        progress.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / total_items, total_correct / total_items


def save_history(history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(figure)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets_by_phase, loaders, class_names = build_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    weights_path = Path(args.weights)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    start_epoch = 1

    if args.resume and weights_path.exists():
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        if "model_state_dict" not in checkpoint:
            raise ValueError("Cannot resume: checkpoint has no training state.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_acc = checkpoint["best_val_accuracy"]
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint.get("history", history)
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_loss, train_acc = run_epoch(
            model, loaders["train"], criterion, optimizer, device, "train", epoch, args.epochs
        )
        val_loss, val_acc = run_epoch(
            model, loaders["val"], criterion, optimizer, device, "val", epoch, args.epochs
        )
        scheduler.step(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        elapsed = time.perf_counter() - started
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | {elapsed:.1f}s"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_accuracy": best_acc,
                    "class_names": class_names,
                    "num_classes": len(class_names),
                    "history": history,
                },
                weights_path,
            )
            print(f"Saved best checkpoint: {weights_path}")

    save_history(history, output_dir)
    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")
    print(f"Training curves saved to {output_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
