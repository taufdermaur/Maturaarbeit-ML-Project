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
                    shuffle=True, 
                    num_workers=0,
                    pin_memory=True)
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
        self.fc1 = nn.Linear(20 * 38 * 38, 500)
        self.fc2 = nn.Linear(500, len(train_dataset.classes))
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 38 * 38)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, optimizer, and loss function
model = TrafficSignClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
loss_function = nn.CrossEntropyLoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n------------------------------------------------------------------------")

# Training loop
def train_model(model, train_loader, optimizer, loss_function, num_epochs=15):
    model.train()
    print("Starting training...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    print("Training complete.\n-----------------------------------------------------------------------")

train_model(model, train_loader, optimizer, loss_function, num_epochs=15)

# Validate the model
def validate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    print("Validating the model...")
    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # Get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test set: {accuracy:.2f}%")
    print(f"Total images: {total}, Correct predictions: {correct}")
    print("Testing complete.\n-----------------------------------------------------------------------")

validate_model(model, test_loader)

# Save the model
path = "models"
model_name = "traffic_sign_classifier.pth"
model_path = f"{path}/{model_name}"

torch.save(obj=model, f= model_path,)
print(f"\nModel saved to {model_path}")


