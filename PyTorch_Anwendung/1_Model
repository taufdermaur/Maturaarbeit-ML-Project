import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
torch.manual_seed(42)

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data
X = "G:"

# Create a train and test split
train_split = int(0.8 * len(X))
X_train = X[:train_split]
X_test = X[train_split:]
print(len(X_train), len(X_test))