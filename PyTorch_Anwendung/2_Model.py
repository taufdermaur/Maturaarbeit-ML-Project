# Import necessary libraries
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

# Set random seed for reproducibility
torch.manual_seed(42)

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define the transformation for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataset
data_dir_train = r"G:\Meine Ablage\Maturaarbeit\Programmierung_Python\First-Steps-ML-Project\dataset_neu\train"
data_dir_test = r"G:\Meine Ablage\Maturaarbeit\Programmierung_Python\First-Steps-ML-Project\dataset_neu\test"


train_dataset = torchvision.datasets.ImageFolder(data_dir_train, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(data_dir_test, transform=transform)

# DataLoader
train_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=32, 
                    num_workers=0)
test_loader = torch.utils.data.DataLoader(
                    test_dataset, 
                    batch_size=32, 
                    num_workers=0)

# Print dataset information
print(f"\nNumber of classes: {len(train_dataset.classes)}")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Model Definition
class TrafficSignClassifier(nn.Module):
    def __init__(self):
        super(TrafficSignClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(20 * 53 * 53, 500) # 20 channels, 53x53 feature map size after two conv layers and pooling
        self.fc2 = nn.Linear(500, len(train_dataset.classes))
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, optimizer, and loss function
model = TrafficSignClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# Training loop
def train_model(model, train_loader, optimizer, loss_function, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

train_model(model, train_loader, optimizer, loss_function, epochs=10)