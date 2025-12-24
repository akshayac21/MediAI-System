import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Define Transforms (Resize is crucial for MRI images of different sizes)
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load Data
try:
    train_dataset = ImageFolder(root='./dataset/Training', transform=data_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    classes = train_dataset.classes
    print(f"Classes found: {classes}")
except FileNotFoundError:
    print("Error: Dataset not found. Please create a 'dataset/Training' folder with your MRI images.")
    exit()

# 4. Define Model
class ResNet18MRI(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18MRI, self).__init__()
        self.resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet18(x)

model = ResNet18MRI(num_classes=len(classes)).to(device)

# 5. Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001) 

# 6. Training Loop
num_epochs = 10  
print("Starting Training...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 7. Save the Model
torch.save(model.state_dict(), "models/brain_tumor_resnet18.pth")
print("Model saved successfully as 'brain_tumor_resnet18.pth'")

print(f"Class mapping: {classes}")