import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# --- Import MedMNIST ---
import medmnist
from medmnist import OCTMNIST
from medmnist import INFO
import os
import numpy as np
from tqdm import tqdm

# 1. Setup Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Data Setup  ---


data_flag = 'octmnist'
download = True

info = INFO[data_flag]
n_classes = len(info['label'])

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load Data (Using MedMNIST)

train_dataset = OCTMNIST(split='train', transform=data_transform, download=download, size=28)
train_dataset.labels = train_dataset.labels.flatten()


BATCH_SIZE = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
classes = [f"Class {i}: {info['label'][str(i)]}" for i in range(n_classes)]

print(f"Classes found: {n_classes} ({', '.join([str(c) for c in classes])})")
print(f"Training dataset size: {len(train_dataset)}")
# 

# --- 4. Define Model (ResNet18) ---
class ResNet18MRI(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18MRI, self).__init__()
        
        self.resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet18(x)

model = ResNet18MRI(num_classes=n_classes).to(device)

# 5. Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 6. Training Loop
NUM_EPOCHS = 10 
print("Starting Training...")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")

    for inputs, labels in progress_bar:
        labels = labels.to(dtype=torch.long)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        current_loss = running_loss / total_samples
        current_acc = 100 * correct_predictions / total_samples
        progress_bar.set_postfix(Loss=f"{current_loss:.4f}", Acc=f"{current_acc:.2f}%")

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100 * correct_predictions / len(train_dataset)

    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} Finished: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}% ---")

# 7. Save the Model
save_path = "models/octmnist_resnet18.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved successfully as '{save_path}'")

print(f"Class mapping: {classes}")