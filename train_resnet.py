import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# ----------- ۱. آماده‌سازی مسیر داده‌ها -----------
train_dir = "data/split/train"
val_dir = "data/split/val"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# ----------- ۲. آماده‌سازی مدل -----------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # نسخه جدید استفاده از weights
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_data.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"استفاده از: {device}")
model = model.to(device)

# ----------- ۳. تعریف معیار خطا و بهینه‌ساز -----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------- ۴. حلقه آموزش -----------
epochs = 1  # می‌تونی بیشتر هم بذاری، ولی ۵ برای شروع خوبه

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%")

# ----------- ۵. ذخیره مدل آموزش‌دیده -----------
# ----------- ۵. ذخیره مدل آموزش‌دیده -----------
torch.save(model.state_dict(), "resnet18_test.pth")
print("✅ مدل ذخیره شد: resnet18_test.pth")