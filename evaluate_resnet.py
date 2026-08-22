import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATA_DIR = os.path.join("data", "split", "test")
MODEL_PATH = os.path.join("weights", "resnet18_best.pth")
OUTPUT_DIR = "outputs"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = dataset.classes

# لود ساختار مدل و وزن‌ها
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
)

# رسم ماتریس آشفتگی (Confusion Matrix)
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
confusion_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(confusion_path, dpi=150)
print(f"Confusion matrix saved as '{confusion_path}'")
