import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

transform = transforms.Compose(
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
)

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5) # 3 input channels (RGB), 6 output channels, 5x5 kernel (filter)


train_folder = r"G:\Meine Ablage\Maturaarbeit\Programmierung_Python\First-Steps-ML-Project\dataset\train"
test_folder = r"G:\Meine Ablage\Maturaarbeit\Programmierung_Python\First-Steps-ML-Project\dataset\test"