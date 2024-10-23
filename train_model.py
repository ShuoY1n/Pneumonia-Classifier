import os
from signal import valid_signals
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class XrayDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for lable in ["NORMAL", "PNEUMONIA"]:
            class_dir = os.path.join(root_dir, lable)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(0 if lable == "NORMAL" else 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        lable = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, lable

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = XrayDataset(root_dir="/mnt/c/Users/jacks/Downloads/archive/chest_xray/train", transform=transform)
test_dataset = XrayDataset(root_dir="/mnt/c/Users/jacks/Downloads/archive/chest_xray/test", transform=transform)
val_dataset = XrayDataset(root_dir="/mnt/c/Users/jacks/Downloads/archive/chest_xray/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

if os.path.exists("pneumonia_model.pth"):
    model.load_state_dict(torch.load("pneumonia_model.pth"))
    print("Model loaded from checkpoint")
else:
    print("Model not found, starting from scratch")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, lables in train_loader:
        images = images.to(device)
        lables = lables.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, lables)
        loss.backward()

        optimizer.step()
        running_loss += loss

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    model.eval()
    val_lables = []
    val_preds = []

    with torch.no_grad():
        for images, lables in val_loader:
            images = images.to(device)
            lables = lables.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_lables.extend(lables.cpu().numpy())
            val_preds.extend(predicted.cpu().numpy())

    val_acc = accuracy_score(val_lables, val_preds)
    print("Validation Accuracy: ", val_acc)

    
model.eval()
test_lables = []
test_preds = []

with torch.no_grad():
    for images, lables in test_loader:
        images = images.to(device)
        lables = lables.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_lables.extend(lables.cpu().numpy())
        test_preds.extend(predicted.cpu().numpy())

test_acc = accuracy_score(test_lables, test_preds)
print("Test Accuracy: ", test_acc)

torch.save(model.state_dict(), "pneumonia_model.pth")