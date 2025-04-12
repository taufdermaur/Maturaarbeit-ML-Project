import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TrafficSignDataset(Dataset):
    def __init__(self, class_1_dir, class_2_dir, transform=None):
        """
        Custom Dataset for traffic sign binary classification
        
        Args:
            class_1_dir (str): Directory with class 1 images
            class_2_dir (str): Directory with class 2 images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Class 1 images (label 0)
        for img_name in os.listdir(class_1_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(class_1_dir, img_name))
                self.labels.append(0)
        
        # Class 2 images (label 1)
        for img_name in os.listdir(class_2_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(class_2_dir, img_name))
                self.labels.append(1)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class SimpleTrafficSignNet(nn.Module):
    def __init__(self):
        super(SimpleTrafficSignNet, self).__init__()
        
        # Simple CNN architecture
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),  # For 64x64 input images
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """
    Train the model
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs
    
    Returns:
        model: Trained model
    """
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float().view(-1, 1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs >= 0.5).float().view(-1)
                total += labels.size(0)
                correct += (predicted == labels.view(-1)).sum().item()
        
        # Print status
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    
    return model

def test_model(model, test_loader):
    """
    Test the model
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
    
    Returns:
        accuracy: Model accuracy on the test set
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predicted = (outputs >= 0.5).float().view(-1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def main(class_1_dir, class_2_dir):
    # Data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((64, 64)),  # Simpler size for our model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Simple normalization
    ])
    
    # Create dataset
    dataset = TrafficSignDataset(class_1_dir, class_2_dir, transform=data_transforms)
    
    # Check if dataset is loaded correctly
    print(f"Dataset size: {len(dataset)}")
    
    # Split dataset into train, validation, and test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize the model
    model = SimpleTrafficSignNet().to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=10
    )
    
    # Save the model
    torch.save(trained_model.state_dict(), 'traffic_sign_model.pth')
    
    # Test the model
    accuracy = test_model(trained_model, test_loader)
    
    return trained_model, accuracy

if __name__ == "__main__":
    # Replace these with your actual directories
    class_1_dir = "path/to/class_1"
    class_2_dir = "path/to/class_2"
    
    # Run the training and evaluation
    model, accuracy = main(class_1_dir, class_2_dir)
