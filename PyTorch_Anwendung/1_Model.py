import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torchvision.io
import torchvision 
import os

# Set random seed for reproducibility
torch.manual_seed(42)

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training data
folder_path = r"G:\Meine Ablage\Maturaarbeit\Programmierung_Python\First-Steps-ML-Project\dataset\train\1_besondere_Wege_und_Busfahrbahn"

image_files = os.listdir(folder_path)

image_tensors = []

for filename in image_files:
    full_path = os.path.join(folder_path, filename)
    with open(full_path, "rb") as f:
        image_bytes = f.read()
    # Bytes in Tensor umwandeln
    image_tensor = torchvision.io.decode_image(torch.tensor(list(image_bytes), dtype=torch.uint8))
    # RGB-Kanäle extrahieren und permutieren (Hight, Width, Channels)
    img_rgb = image_tensor[:3, :, :].permute(1, 2, 0)
    image_tensors.append(img_rgb)

print(f"{len(image_tensors)} Bilder geladen!")

# Model erstellen
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Prepare the training data
import torchvision.transforms as transforms

resize = transforms.Resize((128, 128))  # Alle Bilder auf 128x128 bringen

processed_images = []

for img in image_tensors:
    img_resized = resize(img)           # (H, W, C)
    img_c_hw = img_resized.permute(2, 0, 1)  # (C, H, W)
    processed_images.append(img_c_hw)

batch = torch.stack(processed_images).to(device)  # (Batch, C, H, W)

model = SimpleCNN(num_classes=2).to(device)
outputs = model(batch)  # Forward pass

...