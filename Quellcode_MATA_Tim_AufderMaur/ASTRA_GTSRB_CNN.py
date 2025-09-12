import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import time
import psutil
import GPUtil
from PIL import Image
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Start timer
start_time = time.time()

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Initialize performance monitor
initial_memory = psutil.virtual_memory().percent
cpu_history = []
memory_history = []
gpu_util_history = []
timestamps = []

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Custom Dataset class for GTSRB test data (uses CSV)
class GTSRBTestDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None, class_to_idx=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['Path'])
        image = Image.open(img_path).convert('RGB')
        class_id = self.annotations.iloc[idx]['ClassId']
        
        # Map the actual class ID to the training dataset's index
        if self.class_to_idx is not None:
            folder_name = f"{class_id}_"
            mapped_class = None
            for folder, idx in self.class_to_idx.items():
                if folder.startswith(folder_name):
                    mapped_class = idx
                    break
            
            if mapped_class is not None:
                class_id = mapped_class
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_id

# Load datasets
print("Loading datasets...")

# Load ASTRA dataset
astra_train_dataset = datasets.ImageFolder(
    root=r'C:\Users\timau\Desktop\Datensaetze\ASTRA\Train',
    transform=train_transform
)

astra_test_dataset = datasets.ImageFolder(
    root=r'C:\Users\timau\Desktop\Datensaetze\ASTRA\Test',
    transform=test_transform
)

# Load GTSRB dataset
gtsrb_train_dataset = datasets.ImageFolder(
    root=r'C:\Users\timau\Desktop\Datensaetze\GTSRB\Train',
    transform=train_transform
)

gtsrb_test_dataset = GTSRBTestDataset(
    csv_file=r'C:\Users\timau\Desktop\Datensaetze\GTSRB\Test.csv',
    root_dir=r'C:\Users\timau\Desktop\Datensaetze\GTSRB',
    transform=test_transform,
    class_to_idx=gtsrb_train_dataset.class_to_idx
)

# Dataset information
print("\n=== DATASET INFORMATION ===")
print(f"ASTRA Training: {len(astra_train_dataset)} samples, {len(astra_train_dataset.classes)} classes")
print(f"ASTRA Test: {len(astra_test_dataset)} samples, {len(astra_test_dataset.classes)} classes")
print(f"GTSRB Training: {len(gtsrb_train_dataset)} samples, {len(gtsrb_train_dataset.classes)} classes")
print(f"GTSRB Test: {len(gtsrb_test_dataset)} samples, {len(gtsrb_train_dataset.classes)} classes")

# Train/Validation split for ASTRA
train_size = int(0.8 * len(astra_train_dataset))
val_size = len(astra_train_dataset) - train_size
astra_train_split, astra_val_split = random_split(astra_train_dataset, [train_size, val_size])

astra_train_loader = DataLoader(astra_train_split, batch_size=32, shuffle=True)
astra_val_loader = DataLoader(astra_val_split, batch_size=32, shuffle=False)
astra_test_loader = DataLoader(astra_test_dataset, batch_size=32, shuffle=False)

# Train/Validation split for GTSRB
train_size = int(0.8 * len(gtsrb_train_dataset))
val_size = len(gtsrb_train_dataset) - train_size
gtsrb_train_split, gtsrb_val_split = random_split(gtsrb_train_dataset, [train_size, val_size])

gtsrb_train_loader = DataLoader(gtsrb_train_split, batch_size=32, shuffle=True)
gtsrb_val_loader = DataLoader(gtsrb_val_split, batch_size=32, shuffle=False)
gtsrb_test_loader = DataLoader(gtsrb_test_dataset, batch_size=32, shuffle=False)

# CNN Model Definition
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes):
        super(TrafficSignCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Performance monitoring functions
def get_gpu_usage():
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        return gpu.memoryUtil * 100, gpu.load * 100
    return 0, 0

def log_performance():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    gpu_memory, gpu_util = get_gpu_usage()
    
    cpu_history.append(cpu_percent)
    memory_history.append(memory_percent)
    gpu_util_history.append(gpu_util)
    timestamps.append(time.time())

# Train ASTRA model
print("\n=== ASTRA MODEL TRAINING ===")
astra_num_classes = len(astra_train_dataset.classes)
astra_model = TrafficSignCNN(astra_num_classes).to(device)

# Count parameters
astra_total_params = sum(p.numel() for p in astra_model.parameters())
print(f"ASTRA model parameters: {astra_total_params:,}")

# Training ASTRA model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(astra_model.parameters(), lr=0.001)
best_val_acc = 0.0
num_epochs = 100

astra_train_losses = []
astra_train_accs = []
astra_val_losses = []
astra_val_accs = []

for epoch in range(num_epochs):
    # Log performance every 5 epochs
    if epoch % 5 == 0:
        log_performance()
        
    # Training phase
    astra_model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (data, target) in enumerate(astra_train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = astra_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()
    
    # Validation phase
    astra_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for data, target in astra_val_loader:
            data, target = data.to(device), target.to(device)
            output = astra_model(data)
            loss = criterion(output, target)
            
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()
    
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    avg_train_loss = train_loss / len(astra_train_loader)
    avg_val_loss = val_loss / len(astra_val_loader)

    astra_train_losses.append(avg_train_loss)
    astra_train_accs.append(train_acc)
    astra_val_losses.append(avg_val_loss)
    astra_val_accs.append(val_acc)
    
    print(f'Epoch {epoch+1:3d}/{num_epochs} | Train Acc: {train_acc:6.2f}% | Val Acc: {val_acc:6.2f}% | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc

astra_best_acc = best_val_acc

# Save ASTRA model
torch.save(astra_model.state_dict(), r'C:\Users\timau\Desktop\astra_model.pth')

# Train GTSRB model
print("\n=== GTSRB MODEL TRAINING ===")
gtsrb_num_classes = len(gtsrb_train_dataset.classes)
gtsrb_model = TrafficSignCNN(gtsrb_num_classes).to(device)

# Count parameters
gtsrb_total_params = sum(p.numel() for p in gtsrb_model.parameters())
print(f"GTSRB model parameters: {gtsrb_total_params:,}")

# Training GTSRB model
optimizer = optim.Adam(gtsrb_model.parameters(), lr=0.001)
best_val_acc = 0.0

gtsrb_train_losses = []
gtsrb_train_accs = []
gtsrb_val_losses = []
gtsrb_val_accs = []

for epoch in range(num_epochs):
    # Log performance every 5 epochs
    if epoch % 5 == 0:
        log_performance()
        
    # Training phase
    gtsrb_model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (data, target) in enumerate(gtsrb_train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = gtsrb_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()
    
    # Validation phase
    gtsrb_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for data, target in gtsrb_val_loader:
            data, target = data.to(device), target.to(device)
            output = gtsrb_model(data)
            loss = criterion(output, target)
            
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()
    
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    avg_train_loss = train_loss / len(gtsrb_train_loader)
    avg_val_loss = val_loss / len(gtsrb_val_loader)
    
    gtsrb_train_losses.append(avg_train_loss)
    gtsrb_train_accs.append(train_acc)
    gtsrb_val_losses.append(avg_val_loss)
    gtsrb_val_accs.append(val_acc)
    
    print(f'Epoch {epoch+1:3d}/{num_epochs} | Train Acc: {train_acc:6.2f}% | Val Acc: {val_acc:6.2f}% | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc

gtsrb_best_acc = best_val_acc

# Save GTSRB model
torch.save(gtsrb_model.state_dict(), r'C:\Users\timau\Desktop\gtsrb_model.pth')

# MODEL EVALUATION
print("\n=== MODEL EVALUATION ===")

# ASTRA Model Evaluation
print("\nEvaluating ASTRA model...")
astra_model.eval()
astra_all_predictions = []
astra_all_targets = []
astra_all_confidences = []
astra_class_confidences = {i: [] for i in range(len(astra_train_dataset.classes))}

with torch.no_grad():
    for data, target in astra_test_loader:
        data, target = data.to(device), target.to(device)
        output = astra_model(data)
        probabilities = torch.softmax(output, dim=1)
        confidences, predicted = torch.max(probabilities, 1)
        
        astra_all_predictions.extend(predicted.cpu().numpy())
        astra_all_targets.extend(target.cpu().numpy())
        astra_all_confidences.extend(confidences.cpu().numpy())
        
        # Collect confidences per class
        for i, (pred, conf, true_label) in enumerate(zip(predicted.cpu().numpy(), 
                                                       confidences.cpu().numpy(), 
                                                       target.cpu().numpy())):
            if pred == true_label:  # Only correct predictions
                astra_class_confidences[true_label].append(conf)

astra_accuracy = accuracy_score(astra_all_targets, astra_all_predictions)
astra_precision, astra_recall, astra_f1, _ = precision_recall_fscore_support(astra_all_targets, astra_all_predictions, average='weighted')

# Calculate average confidence per class for ASTRA
astra_avg_class_confidences = {}
for class_id, confs in astra_class_confidences.items():
    if confs:
        astra_avg_class_confidences[class_id] = np.mean(confs)
    else:
        astra_avg_class_confidences[class_id] = 0.0

print("ASTRA Results:")
print(f"  Accuracy:  {astra_accuracy:.4f}")
print(f"  Precision: {astra_precision:.4f}")
print(f"  Recall:    {astra_recall:.4f}")
print(f"  F1-Score:  {astra_f1:.4f}")

# GTSRB Model Evaluation
print("\nEvaluating GTSRB model...")
gtsrb_model.eval()
gtsrb_all_predictions = []
gtsrb_all_targets = []
gtsrb_all_confidences = []
gtsrb_class_confidences = {i: [] for i in range(len(gtsrb_train_dataset.classes))}

with torch.no_grad():
    for data, target in gtsrb_test_loader:
        data, target = data.to(device), target.to(device)
        output = gtsrb_model(data)
        probabilities = torch.softmax(output, dim=1)
        confidences, predicted = torch.max(probabilities, 1)
        
        gtsrb_all_predictions.extend(predicted.cpu().numpy())
        gtsrb_all_targets.extend(target.cpu().numpy())
        gtsrb_all_confidences.extend(confidences.cpu().numpy())
        
        # Collect confidences per class
        for i, (pred, conf, true_label) in enumerate(zip(predicted.cpu().numpy(), 
                                                       confidences.cpu().numpy(), 
                                                       target.cpu().numpy())):
            if pred == true_label:  # Only correct predictions
                gtsrb_class_confidences[true_label].append(conf)

gtsrb_accuracy = accuracy_score(gtsrb_all_targets, gtsrb_all_predictions)
gtsrb_precision, gtsrb_recall, gtsrb_f1, _ = precision_recall_fscore_support(gtsrb_all_targets, gtsrb_all_predictions, average='weighted')

# Calculate average confidence per class for GTSRB
gtsrb_avg_class_confidences = {}
for class_id, confs in gtsrb_class_confidences.items():
    if confs:
        gtsrb_avg_class_confidences[class_id] = np.mean(confs)
    else:
        gtsrb_avg_class_confidences[class_id] = 0.0

print("GTSRB Results:")
print(f"  Accuracy:  {gtsrb_accuracy:.4f}")
print(f"  Precision: {gtsrb_precision:.4f}")
print(f"  Recall:    {gtsrb_recall:.4f}")
print(f"  F1-Score:  {gtsrb_f1:.4f}")



# Confidence Analysis
print("\n=== CONFIDENCE ANALYSIS ===")

print(f"\nASTRA - Average confidence per class:")
astra_conf_sorted = sorted(astra_avg_class_confidences.items())
for class_id, confidence in astra_conf_sorted:
    print(f"Class {class_id:2d}: {confidence:.4f}")

print(f"\nGTSRB - Average confidence per class:")
gtsrb_conf_sorted = sorted(gtsrb_avg_class_confidences.items())
for class_id, confidence in gtsrb_conf_sorted:
    print(f"Class {class_id:2d}: {confidence:.4f}")

# Create visualizations
print("\n=== CREATING VISUALIZATIONS ===")

# Create confusion matrix for ASTRA
astra_cm = confusion_matrix(astra_all_targets, astra_all_predictions)

plt.figure(figsize=(12, 10))
plt.imshow(astra_cm, interpolation='nearest', cmap='Blues')
plt.title('Konfusionsmatrix - ASTRA', fontsize=18, fontweight='bold', pad=20)
plt.colorbar(label='Anzahl der Vorhersagen')

# Add text annotations
thresh = astra_cm.max() / 2.
for i in range(astra_cm.shape[0]):
    for j in range(astra_cm.shape[1]):
        plt.text(j, i, format(astra_cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if astra_cm[i, j] > thresh else "black",
                fontsize=10)

# Set labels with class numbers only
tick_marks = np.arange(len(astra_train_dataset.classes))
plt.xticks(tick_marks, range(len(astra_train_dataset.classes)), rotation=45, fontsize=12)
plt.yticks(tick_marks, range(len(astra_train_dataset.classes)), fontsize=12)

plt.ylabel('Echte Klasse', fontweight='bold', fontsize=14)
plt.xlabel('Vorhergesagte Klasse', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(r'C:\Users\timau\Desktop\astra_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Create confusion matrix for GTSRB
gtsrb_cm = confusion_matrix(gtsrb_all_targets, gtsrb_all_predictions)

plt.figure(figsize=(12, 10))
plt.imshow(gtsrb_cm, interpolation='nearest', cmap='Blues')
plt.title('Konfusionsmatrix - GTSRB', fontsize=18, fontweight='bold', pad=20)
plt.colorbar(label='Anzahl der Vorhersagen')

# Add text annotations
thresh = gtsrb_cm.max() / 2.
for i in range(gtsrb_cm.shape[0]):
    for j in range(gtsrb_cm.shape[1]):
        plt.text(j, i, format(gtsrb_cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if gtsrb_cm[i, j] > thresh else "black",
                fontsize=8)

# Set labels with class numbers only
tick_marks = np.arange(len(gtsrb_train_dataset.classes))
plt.xticks(tick_marks, range(len(gtsrb_train_dataset.classes)), rotation=45, fontsize=12)
plt.yticks(tick_marks, range(len(gtsrb_train_dataset.classes)), fontsize=12)

plt.ylabel('Echte Klasse', fontweight='bold', fontsize=14)
plt.xlabel('Vorhergesagte Klasse', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(r'C:\Users\timau\Desktop\gtsrb_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("Confusion matrices created.")

# Create training history plots
print("Creating training history plots...")
print(f"ASTRA trained for {len(astra_train_accs)} epochs")
print(f"GTSRB trained for {len(gtsrb_train_accs)} epochs")

astra_epochs = range(1, len(astra_train_accs) + 1)
gtsrb_epochs = range(1, len(gtsrb_train_accs) + 1)

# ASTRA Training History Plot with dual y-axis
epochs = range(1, num_epochs + 1)

fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

# Plot ASTRA accuracy curves on left y-axis
ax1.plot(astra_epochs, astra_train_accs, 'b-', label='Trainings-Genauigkeit', linewidth=2)
ax1.plot(astra_epochs, astra_val_accs, 'b:', label='Validierungs-Genauigkeit', linewidth=2)

ax1.set_xlabel('Epoche', fontsize=16)
ax1.set_ylabel('Genauigkeit (%)', fontsize=16, color='black')
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3)

# Create second y-axis for ASTRA loss
ax2 = ax1.twinx()
ax2.plot(astra_epochs, astra_train_losses, 'r-', label='Trainings-Verlust', linewidth=2)
ax2.plot(astra_epochs, astra_val_losses, 'r:', label='Validierungs-Verlust', linewidth=2)

ax2.set_ylabel('Verlust', fontsize=16, color='gray')
ax2.tick_params(axis='y', labelsize=14, colors='gray')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc='center right')

ax1.set_title('Trainingsverlauf - ASTRA', fontweight='bold', fontsize=18, pad=20)

plt.tight_layout()
plt.savefig(r'C:\Users\timau\Desktop\astra_training_history.png', dpi=300, bbox_inches='tight')
plt.close()

# GTSRB Training History Plot with dual y-axis
fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

# Plot GTSRB accuracy curves on left y-axis
ax1.plot(gtsrb_epochs, gtsrb_train_accs, 'b-', label='Trainings-Genauigkeit', linewidth=2)
ax1.plot(gtsrb_epochs, gtsrb_val_accs, 'b:', label='Validierungs-Genauigkeit', linewidth=2)

ax1.set_xlabel('Epoche', fontsize=16)
ax1.set_ylabel('Genauigkeit (%)', fontsize=16, color='black')
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3)

# Create second y-axis for GTSRB loss
ax2 = ax1.twinx()
ax2.plot(gtsrb_epochs, gtsrb_train_losses, 'r-', label='Trainings-Verlust', linewidth=2, alpha=0.7)
ax2.plot(gtsrb_epochs, gtsrb_val_losses, 'r:', label='Validierungs-Verlust', linewidth=2, alpha=0.7)

ax2.set_ylabel('Verlust', fontsize=16, color='gray')
ax2.tick_params(axis='y', labelsize=14, colors='gray')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc='center right')

ax1.set_title('Trainingsverlauf - GTSRB', fontweight='bold', fontsize=18, pad=20)

plt.tight_layout()
plt.savefig(r'C:\Users\timau\Desktop\gtsrb_training_history.png', dpi=300, bbox_inches='tight')
plt.close()

print("Training history plots created.")

# Real-time performance evaluation
print("\n=== REAL-TIME PERFORMANCE EVALUATION ===")

# ASTRA real-time evaluation
print("Measuring ASTRA real-time latency...")
astra_model.eval()

# Warmup phase
with torch.no_grad():
    warmup_batches = 10
    for i, (data, _) in enumerate(astra_test_loader):
        if i >= warmup_batches:
            break
        data = data.to(device)
        _ = astra_model(data)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

# Realistic single-image latency measurement
astra_per_image_latencies = []
astra_total_samples = 0
astra_total_time = 0
max_samples = 1000  # Limit for reasonable test duration

with torch.no_grad():
    for batch_data, _ in astra_test_loader:
        # Process each image individually for realistic latency
        for single_image in batch_data:
            if astra_total_samples >= max_samples:
                break
            
            # Single image processing (batch_size = 1)
            single_image = single_image.unsqueeze(0).to(device)
            
            # Measure time for single image
            start_time_single = time.time()
            _ = astra_model(single_image)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time_single = time.time()
            
            # Calculate latency in milliseconds
            latency_ms = (end_time_single - start_time_single) * 1000
            astra_per_image_latencies.append(latency_ms)
            
            astra_total_time += (end_time_single - start_time_single)
            astra_total_samples += 1
        
        if astra_total_samples >= max_samples:
            break

# Calculate throughput
astra_throughput = astra_total_samples / astra_total_time if astra_total_time > 0 else 0  # samples per second

# GTSRB real-time evaluation
print("Measuring GTSRB real-time latency...")
gtsrb_model.eval()

# Warmup phase
with torch.no_grad():
    for i, (data, _) in enumerate(gtsrb_test_loader):
        if i >= warmup_batches:
            break
        data = data.to(device)
        _ = gtsrb_model(data)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

# Realistic single-image latency measurement
gtsrb_per_image_latencies = []
gtsrb_total_samples = 0
gtsrb_total_time = 0

with torch.no_grad():
    for batch_data, _ in gtsrb_test_loader:
        # Process each image individually for realistic latency
        for single_image in batch_data:
            if gtsrb_total_samples >= max_samples:
                break
            
            # Single image processing (batch_size = 1)
            single_image = single_image.unsqueeze(0).to(device)
            
            # Measure time for single image
            start_time_single = time.time()
            _ = gtsrb_model(single_image)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time_single = time.time()
            
            # Calculate latency in milliseconds
            latency_ms = (end_time_single - start_time_single) * 1000
            gtsrb_per_image_latencies.append(latency_ms)
            
            gtsrb_total_time += (end_time_single - start_time_single)
            gtsrb_total_samples += 1
        
        if gtsrb_total_samples >= max_samples:
            break

# Calculate throughput
gtsrb_throughput = gtsrb_total_samples / gtsrb_total_time if gtsrb_total_time > 0 else 0  # samples per second

# Create latency histograms for ASTRA
print("Creating latency histograms...")

# ASTRA latency histograms
astra_mean_latency = np.mean(astra_per_image_latencies)
astra_median_latency = np.median(astra_per_image_latencies)
astra_std_latency = np.std(astra_per_image_latencies)

# Separate latencies <= 4ms and > 4ms
astra_latencies_filtered = [lat for lat in astra_per_image_latencies if lat <= 4.0]
astra_latencies_over_4ms = [lat for lat in astra_per_image_latencies if lat > 4.0]

print(f"ASTRA: {len(astra_latencies_over_4ms)} of {len(astra_per_image_latencies)} samples over 4ms ({len(astra_latencies_over_4ms)/len(astra_per_image_latencies)*100:.1f}%)")

# Create bins: 0-4ms in regular intervals, plus one bin for >4ms
regular_bins = np.linspace(0, 4, 40)  # 39 bins from 0-4ms

# Histogram WITHOUT statistics
plt.figure(figsize=(14, 10))

# Create histogram for ≤4ms data
plt.hist(astra_latencies_filtered, bins=regular_bins, alpha=0.7, 
        color='steelblue', edgecolor='black')

# Add the >4ms bin manually
if len(astra_latencies_over_4ms) > 0:
    bin_width = regular_bins[1] - regular_bins[0]
    plt.bar(4.0, len(astra_latencies_over_4ms), width=bin_width, 
           alpha=0.7, color='red', edgecolor='black')

# Set y-axis to log scale
plt.yscale('log')

plt.title('Latenzverteilung - ASTRA', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Latenz pro Bild (ms)', fontweight='bold', fontsize=14)
plt.ylabel('Häufigkeit', fontweight='bold', fontsize=14)

# Set x-axis limits and ticks
plt.xlim(0, 4.2)
xticks = list(np.arange(0, 4.5, 0.5))
xtick_labels = [f'{x:.1f}' if x <= 4.0 else '>4' for x in xticks]
xtick_labels[-1] = '>4'  # Ensure last label is '>4'
plt.xticks(xticks, xtick_labels, fontsize=12)
plt.yticks(fontsize=12)

plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\timau\Desktop\astra_latency_histogram.png', dpi=300, bbox_inches='tight')
plt.close()

# GTSRB latency histograms
gtsrb_mean_latency = np.mean(gtsrb_per_image_latencies)
gtsrb_median_latency = np.median(gtsrb_per_image_latencies)
gtsrb_std_latency = np.std(gtsrb_per_image_latencies)

# Separate latencies <= 4ms and > 4ms
gtsrb_latencies_filtered = [lat for lat in gtsrb_per_image_latencies if lat <= 4.0]
gtsrb_latencies_over_4ms = [lat for lat in gtsrb_per_image_latencies if lat > 4.0]

print(f"GTSRB: {len(gtsrb_latencies_over_4ms)} of {len(gtsrb_per_image_latencies)} samples over 4ms ({len(gtsrb_latencies_over_4ms)/len(gtsrb_per_image_latencies)*100:.1f}%)")

# Histogram WITHOUT statistics
plt.figure(figsize=(14, 10))

# Create histogram for ≤4ms data
plt.hist(gtsrb_latencies_filtered, bins=regular_bins, alpha=0.7, 
        color='steelblue', edgecolor='black')

# Add the >4ms bin manually
if len(gtsrb_latencies_over_4ms) > 0:
    bin_width = regular_bins[1] - regular_bins[0]
    plt.bar(4.0, len(gtsrb_latencies_over_4ms), width=bin_width, 
           alpha=0.7, color='red', edgecolor='black')

# Set y-axis to log scale
plt.yscale('log')

plt.title('Latenzverteilung - GTSRB', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Latenz pro Bild (ms)', fontweight='bold', fontsize=14)
plt.ylabel('Häufigkeit)', fontweight='bold', fontsize=14)

# Set x-axis limits and ticks
plt.xlim(0, 4.2)
xticks = list(np.arange(0, 4.5, 0.5))
xtick_labels = [f'{x:.1f}' if x <= 4.0 else '>4' for x in xticks]
xtick_labels[-1] = '>4'  # Ensure last label is '>4'
plt.xticks(xticks, xtick_labels, fontsize=12)
plt.yticks(fontsize=12)

plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\timau\Desktop\gtsrb_latency_histogram.png', dpi=300, bbox_inches='tight')
plt.close()

print("Latency histograms created.")

# Print real-time performance summary
print("\n=== REAL-TIME PERFORMANCE SUMMARY ===")
print(f"ASTRA: {astra_mean_latency:.2f} ms latency, {astra_throughput:.1f} samples/s throughput")
print(f"GTSRB: {gtsrb_mean_latency:.2f} ms latency, {gtsrb_throughput:.1f} samples/s throughput")

# Create performance plots
print("\nCreating system performance plots...")

# Convert timestamps to relative time in minutes
start_time_monitor = timestamps[0]
time_minutes = [(t - start_time_monitor) / 60 for t in timestamps]

# Create time series plot (4 subplots)
fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(30, 10))
fig.suptitle('Systemleistung während des Trainings', fontsize=20, fontweight='bold')

# CPU Usage
ax1.plot(time_minutes, cpu_history, color='#2E86C1', linewidth=2)
ax1.set_title('CPU-Auslastung', fontweight='bold', fontsize=16)
ax1.set_xlabel('Zeit in Minuten', fontsize=14)
ax1.set_ylabel('CPU-Auslastung (%)', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 100)
ax1.tick_params(axis='both', which='major', labelsize=12)

# Memory Usage
ax2.plot(time_minutes, memory_history, color='#E74C3C', linewidth=2)
ax2.set_title('Arbeitsspeicher-Auslastung', fontweight='bold', fontsize=16)
ax2.set_xlabel('Zeit in Minuten', fontsize=14)
ax2.set_ylabel('RAM-Auslastung (%)', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 100)
ax2.tick_params(axis='both', which='major', labelsize=12)

# GPU Utilization
ax3.plot(time_minutes, gpu_util_history, color='#F39C12', linewidth=2)
ax3.set_title('Grafikkarte-Auslastung', fontweight='bold', fontsize=16)
ax3.set_xlabel('Zeit in Minuten', fontsize=14)
ax3.set_ylabel('GPU-Auslastung (%)', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 100)
ax3.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.savefig(r'C:\Users\timau\Desktop\systemleistung.png', dpi=300, bbox_inches='tight')
plt.close()

# Create summary statistics plot
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

categories = ['CPU', 'RAM', 'GPU-Auslastung']
averages = [
    np.mean(cpu_history),
    np.mean(memory_history),
    np.mean(gpu_util_history)
]
maxima = [
    np.max(cpu_history),
    np.max(memory_history),
    np.max(gpu_util_history)
]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, averages, width, label='Durchschnitt', color='#3498DB', alpha=0.8)
bars2 = ax.bar(x + width/2, maxima, width, label='Maximum', color='#E74C3C', alpha=0.8)

ax.set_ylabel('Auslastung (%)', fontsize=16)
ax.set_title('Systemleistung - Zusammenfassung', fontweight='bold', fontsize=18, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='y', labelsize=14)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(r'C:\Users\timau\Desktop\systemleistung_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("System performance plots created.")

# Total execution time
total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time/60:.2f} minutes")
print("\nSaved files:")
print("- systemleistung.png")
print("- systemleistung_summary.png")
print("- astra_confusion_matrix.png")
print("- gtsrb_confusion_matrix.png")
print("- astra_training_history.png")
print("- gtsrb_training_history.png")
print("- astra_latency_histogram.png")
print("- gtsrb_latency_histogram.png")
print("- astra_model.pth")
print("- gtsrb_model.pth")

print("\n=== SCRIPT BEENDET ===")