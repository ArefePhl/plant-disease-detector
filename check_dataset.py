"""Run basic integrity checks on train/val/test ImageFolder datasets."""

from pathlib import Path

from torchvision.datasets import ImageFolder


ROOT = Path("data/split")
PHASES = ("train", "val", "test")


def main():
    datasets = {phase: ImageFolder(ROOT / phase) for phase in PHASES}
    class_names = datasets["train"].classes
    for phase, dataset in datasets.items():
        if dataset.classes != class_names:
            raise ValueError(f"Class mismatch in {phase}")
        print(f"{phase}: {len(dataset)} images, {len(dataset.classes)} classes")

    paths_by_phase = {
        phase: {Path(path).resolve() for path, _ in dataset.samples}
        for phase, dataset in datasets.items()
    }
    for left_index, left in enumerate(PHASES):
        for right in PHASES[left_index + 1:]:
            overlap = paths_by_phase[left] & paths_by_phase[right]
            if overlap:
                raise ValueError(f"Overlapping files: {left} and {right}")
    print("Dataset integrity checks passed.")


if __name__ == "__main__":
    main()
