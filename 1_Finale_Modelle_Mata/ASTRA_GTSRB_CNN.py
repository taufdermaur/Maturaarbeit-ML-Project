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

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.initial_memory = psutil.virtual_memory().percent
        self.cpu_history = []
        self.memory_history = []
        self.gpu_memory_history = []
        self.gpu_util_history = []
        self.timestamps = []
        
    def get_gpu_usage(self):
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            return gpu.memoryUtil * 100, gpu.load * 100
        return 0, 0
    
    def log_performance(self):
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        gpu_memory, gpu_util = self.get_gpu_usage()
        
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_percent)
        self.gpu_memory_history.append(gpu_memory)
        self.gpu_util_history.append(gpu_util)
        self.timestamps.append(time.time())
    
    def create_performance_plots(self, save_path):
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8')
        
        # Convert timestamps to relative time in minutes
        start_time = self.timestamps[0]
        time_minutes = [(t - start_time) / 60 for t in self.timestamps]
        
        # Create time series plot (4 subplots)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Systemleistung während des Trainings', fontsize=16, fontweight='bold')
        
        # CPU Usage
        ax1.plot(time_minutes, self.cpu_history, color='#2E86C1', linewidth=2)
        ax1.set_title('CPU-Auslastung', fontweight='bold')
        ax1.set_xlabel('Zeit (Minuten)')
        ax1.set_ylabel('CPU-Auslastung (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Memory Usage
        ax2.plot(time_minutes, self.memory_history, color='#E74C3C', linewidth=2)
        ax2.set_title('RAM-Auslastung', fontweight='bold')
        ax2.set_xlabel('Zeit (Minuten)')
        ax2.set_ylabel('RAM-Auslastung (%)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # GPU Memory Usage
        ax3.plot(time_minutes, self.gpu_memory_history, color='#28B463', linewidth=2)
        ax3.set_title('GPU-Speicher-Auslastung', fontweight='bold')
        ax3.set_xlabel('Zeit (Minuten)')
        ax3.set_ylabel('GPU-Speicher (%)')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # GPU Utilization
        ax4.plot(time_minutes, self.gpu_util_history, color='#F39C12', linewidth=2)
        ax4.set_title('GPU-Auslastung', fontweight='bold')
        ax4.set_xlabel('Zeit (Minuten)')
        ax4.set_ylabel('GPU-Auslastung (%)')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        categories = ['CPU', 'RAM', 'GPU-Speicher', 'GPU-Auslastung']
        averages = [
            np.mean(self.cpu_history),
            np.mean(self.memory_history),
            np.mean(self.gpu_memory_history),
            np.mean(self.gpu_util_history)
        ]
        maxima = [
            np.max(self.cpu_history),
            np.max(self.memory_history),
            np.max(self.gpu_memory_history),
            np.max(self.gpu_util_history)
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, averages, width, label='Durchschnitt', color='#3498DB', alpha=0.8)
        bars2 = ax.bar(x + width/2, maxima, width, label='Maximum', color='#E74C3C', alpha=0.8)
        
        ax.set_ylabel('Auslastung (%)')
        ax.set_title('Systemleistung - Zusammenfassung', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# Start timer
start_time = time.time()

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Initialize performance monitor
monitor = PerformanceMonitor()

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
print("Lade Datensätze...")

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
print("\n=== DATENSATZ INFORMATIONEN ===")
print(f"ASTRA Training: {len(astra_train_dataset)} Samples, {len(astra_train_dataset.classes)} Klassen")
print(f"ASTRA Test: {len(astra_test_dataset)} Samples, {len(astra_test_dataset.classes)} Klassen")
print(f"GTSRB Training: {len(gtsrb_train_dataset)} Samples, {len(gtsrb_train_dataset.classes)} Klassen")
print(f"GTSRB Test: {len(gtsrb_test_dataset)} Samples, {len(gtsrb_train_dataset.classes)} Klassen")

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

# Training function
def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, monitor=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Log performance every 5 epochs
        if monitor and epoch % 5 == 0:
            monitor.log_performance()
            
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
    return best_val_acc

# Evaluation function
def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_predictions = []
    all_targets = []
    all_confidences = []
    class_confidences = {i: [] for i in range(len(class_names))}
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            
            # Collect confidences per class
            for i, (pred, conf, true_label) in enumerate(zip(predicted.cpu().numpy(), 
                                                           confidences.cpu().numpy(), 
                                                           target.cpu().numpy())):
                if pred == true_label:  # Only correct predictions
                    class_confidences[true_label].append(conf)
    
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')
    
    # Calculate average confidence per class
    avg_class_confidences = {}
    for class_id, confs in class_confidences.items():
        if confs:
            avg_class_confidences[class_id] = np.mean(confs)
        else:
            avg_class_confidences[class_id] = 0.0
    
    return accuracy, precision, recall, f1, all_predictions, all_targets, avg_class_confidences

# Function to create confusion matrix
def create_confusion_matrix(y_true, y_pred, num_classes, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Konfusionsmatrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.colorbar(label='Anzahl Vorhersagen')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
    
    # Set labels with class numbers only
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes), rotation=45)
    plt.yticks(tick_marks, range(num_classes))
    
    plt.ylabel('Wahre Klasse', fontweight='bold')
    plt.xlabel('Vorhergesagte Klasse', fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Function to create confidence analysis plot
def create_confidence_plot(confidences, model_name, save_path):
    classes = list(confidences.keys())
    conf_values = [confidences[c] for c in classes]
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(classes, conf_values, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for bar, conf in zip(bars, conf_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'Durchschnittliche Konfidenz pro Klasse - {model_name}', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Klasse', fontweight='bold')
    plt.ylabel('Durchschnittliche Konfidenz', fontweight='bold')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels if there are many classes
    if len(classes) > 20:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Real-time evaluation function
def realtime_evaluation_detailed(model, test_loader, model_name, warmup_batches=10):
    model.eval()
    
    # Warmup phase
    print(f"Warmup für {model_name}...")
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= warmup_batches:
                break
            data = data.to(device)
            _ = model(data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Realistic single-image latency measurement
    print(f"Messe realistische Einzelbild-Latenz für {model_name}...")
    per_image_latencies = []
    total_samples = 0
    total_time = 0
    max_samples = 1000  # Limit for reasonable test duration
    
    with torch.no_grad():
        for batch_data, _ in test_loader:
            # Process each image individually for realistic latency
            for single_image in batch_data:
                if total_samples >= max_samples:
                    break
                
                # Single image processing (batch_size = 1)
                single_image = single_image.unsqueeze(0).to(device)
                
                # Measure time for single image
                start_time = time.time()
                _ = model(single_image)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                # Calculate latency in milliseconds
                latency_ms = (end_time - start_time) * 1000
                per_image_latencies.append(latency_ms)
                
                total_time += (end_time - start_time)
                total_samples += 1
            
            if total_samples >= max_samples:
                break
    
    # Calculate throughput
    throughput = total_samples / total_time if total_time > 0 else 0  # samples per second
    
    return per_image_latencies, throughput

# Function to create latency histogram with modified axes
def create_latency_histogram(latencies, model_name, save_path_with_stats, save_path_without_stats):
    mean_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    std_latency = np.std(latencies)
    
    # Separate latencies <= 4ms and > 4ms
    latencies_filtered = [lat for lat in latencies if lat <= 4.0]
    latencies_over_4ms = [lat for lat in latencies if lat > 4.0]
    
    print(f"{model_name}: {len(latencies_over_4ms)} von {len(latencies)} Samples über 4ms ({len(latencies_over_4ms)/len(latencies)*100:.1f}%)")
    
    # Create bins: 0-4ms in regular intervals, plus one bin for >4ms
    regular_bins = np.linspace(0, 4, 40)  # 39 bins from 0-4ms
    
    # Histogram WITH statistics
    plt.figure(figsize=(12, 8))
    
    # Create histogram for ≤4ms data
    n, bins, patches = plt.hist(latencies_filtered, bins=regular_bins, alpha=0.7, 
                               color='steelblue', edgecolor='black')
    
    # Add the >4ms bin manually
    if len(latencies_over_4ms) > 0:
        # Get the width of regular bins for consistent appearance
        bin_width = regular_bins[1] - regular_bins[0]
        
        # Add the >4ms bar at position 4.0
        plt.bar(4.0, len(latencies_over_4ms), width=bin_width, 
               alpha=0.7, color='red', edgecolor='black', label=f'>4ms (n={len(latencies_over_4ms)})')
    
    # Set y-axis to log scale
    plt.yscale('log')
    
    # Add vertical lines for mean and median (only if they're ≤4ms)
    if mean_latency <= 4.0:
        plt.axvline(mean_latency, color='red', linestyle='--', linewidth=2, 
                   label=f'Mittelwert: {mean_latency:.2f} ms')
    if median_latency <= 4.0:
        plt.axvline(median_latency, color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {median_latency:.2f} ms')
    
    plt.title(f'Latenz-Verteilung - {model_name} (mit Statistiken)', fontsize=16, fontweight='bold')
    plt.xlabel('Latenz pro Bild (ms)', fontweight='bold')
    plt.ylabel('Häufigkeit (log scale)', fontweight='bold')
    
    # Set x-axis limits and ticks
    plt.xlim(0, 4.2)
    xticks = list(np.arange(0, 4.5, 0.5))
    xtick_labels = [f'{x:.1f}' if x <= 4.0 else '>4' for x in xticks]
    xtick_labels[-1] = '>4'  # Ensure last label is '>4'
    plt.xticks(xticks, xtick_labels)
    
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add statistics text box
    over_4ms_percent = len(latencies_over_4ms)/len(latencies)*100
    stats_text = (f'Statistiken:\n'
                 f'Mittelwert: {mean_latency:.2f} ms\n'
                 f'Median: {median_latency:.2f} ms\n'
                 f'Std.-Abw.: {std_latency:.2f} ms\n'
                 f'Min: {min(latencies):.2f} ms\n'
                 f'Max: {max(latencies):.2f} ms\n'
                 f'>4ms: {over_4ms_percent:.1f}% ({len(latencies_over_4ms)} samples)')
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path_with_stats, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Histogram WITHOUT statistics
    plt.figure(figsize=(12, 8))
    
    # Create histogram for ≤4ms data
    plt.hist(latencies_filtered, bins=regular_bins, alpha=0.7, 
            color='steelblue', edgecolor='black')
    
    # Add the >4ms bin manually
    if len(latencies_over_4ms) > 0:
        bin_width = regular_bins[1] - regular_bins[0]
        plt.bar(4.0, len(latencies_over_4ms), width=bin_width, 
               alpha=0.7, color='red', edgecolor='black')
    
    # Set y-axis to log scale
    plt.yscale('log')
    
    plt.title(f'Latenz-Verteilung - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Latenz pro Bild (ms)', fontweight='bold')
    plt.ylabel('Häufigkeit (log scale)', fontweight='bold')
    
    # Set x-axis limits and ticks
    plt.xlim(0, 4.2)
    xticks = list(np.arange(0, 4.5, 0.5))
    xtick_labels = [f'{x:.1f}' if x <= 4.0 else '>4' for x in xticks]
    xtick_labels[-1] = '>4'  # Ensure last label is '>4'
    plt.xticks(xticks, xtick_labels)
    
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path_without_stats, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mean_latency

# Train ASTRA model
print("\n=== ASTRA MODELL TRAINING ===")
astra_num_classes = len(astra_train_dataset.classes)
astra_model = TrafficSignCNN(astra_num_classes).to(device)

# Count parameters
astra_total_params = sum(p.numel() for p in astra_model.parameters())
print(f"ASTRA Modell Parameter: {astra_total_params:,}")

astra_best_acc = train_model(astra_model, astra_train_loader, astra_val_loader, num_epochs=100, monitor=monitor)

# Save ASTRA model
torch.save(astra_model.state_dict(), r'C:\Users\timau\Desktop\astra_model.pth')

# Train GTSRB model
print("\n=== GTSRB MODELL TRAINING ===")
gtsrb_num_classes = len(gtsrb_train_dataset.classes)
gtsrb_model = TrafficSignCNN(gtsrb_num_classes).to(device)

# Count parameters
gtsrb_total_params = sum(p.numel() for p in gtsrb_model.parameters())
print(f"GTSRB Modell Parameter: {gtsrb_total_params:,}")

gtsrb_best_acc = train_model(gtsrb_model, gtsrb_train_loader, gtsrb_val_loader, num_epochs=100, monitor=monitor)

# Save GTSRB model
torch.save(gtsrb_model.state_dict(), r'C:\Users\timau\Desktop\gtsrb_model.pth')

# Evaluation
print("\n=== MODELL EVALUATION ===")

# ASTRA evaluation
astra_accuracy, astra_precision, astra_recall, astra_f1, astra_predictions, astra_targets, astra_confidences = evaluate_model(
    astra_model, astra_test_loader, astra_train_dataset.classes
)

# GTSRB evaluation
gtsrb_accuracy, gtsrb_precision, gtsrb_recall, gtsrb_f1, gtsrb_predictions, gtsrb_targets, gtsrb_confidences = evaluate_model(
    gtsrb_model, gtsrb_test_loader, gtsrb_train_dataset.classes
)

# Results table
results_data = {
    'Model': ['ASTRA', 'GTSRB'],
    'Accuracy': [f'{astra_accuracy:.4f}', f'{gtsrb_accuracy:.4f}'],
    'Precision': [f'{astra_precision:.4f}', f'{gtsrb_precision:.4f}'],
    'Recall': [f'{astra_recall:.4f}', f'{gtsrb_recall:.4f}'],
    'F1-Score': [f'{astra_f1:.4f}', f'{gtsrb_f1:.4f}']
}

results_df = pd.DataFrame(results_data)
print("\nEvaluierungsergebnisse:")
print(results_df.to_string(index=False))

# Confidence Analysis
print("\n=== KONFIDENZ ANALYSE ===")

print(f"\nASTRA - Durchschnittliche Konfidenz pro Klasse:")
astra_conf_sorted = sorted(astra_confidences.items())
for class_id, confidence in astra_conf_sorted:
    print(f"Klasse {class_id}: {confidence:.4f}")

print(f"\nGTSRB - Durchschnittliche Konfidenz pro Klasse:")
gtsrb_conf_sorted = sorted(gtsrb_confidences.items())
for class_id, confidence in gtsrb_conf_sorted:
    print(f"Klasse {class_id}: {confidence:.4f}")

# Create visualizations
print("\n=== ERSTELLE VISUALISIERUNGEN ===")

# Confusion matrices
print("Erstelle Konfusionsmatrizen...")
create_confusion_matrix(astra_targets, astra_predictions, 
                       len(astra_train_dataset.classes), 
                       "ASTRA", 
                       r'C:\Users\timau\Desktop\astra_confusion_matrix.png')

create_confusion_matrix(gtsrb_targets, gtsrb_predictions, 
                       len(gtsrb_train_dataset.classes), 
                       "GTSRB", 
                       r'C:\Users\timau\Desktop\gtsrb_confusion_matrix.png')

# Real-time performance evaluation
print("\n=== ECHTZEIT-PERFORMANCE EVALUATION ===")

# ASTRA real-time evaluation
astra_latencies, astra_throughput = realtime_evaluation_detailed(
    astra_model, astra_test_loader, "ASTRA"
)

# GTSRB real-time evaluation  
gtsrb_latencies, gtsrb_throughput = realtime_evaluation_detailed(
    gtsrb_model, gtsrb_test_loader, "GTSRB"
)

# Create latency histograms
print("Erstelle Latenz-Histogramme...")
astra_mean_lat = create_latency_histogram(
    astra_latencies, "ASTRA",
    r'C:\Users\timau\Desktop\astra_latency_histogram_mit_statistiken.png',
    r'C:\Users\timau\Desktop\astra_latency_histogram.png'
)

gtsrb_mean_lat = create_latency_histogram(
    gtsrb_latencies, "GTSRB", 
    r'C:\Users\timau\Desktop\gtsrb_latency_histogram_mit_statistiken.png',
    r'C:\Users\timau\Desktop\gtsrb_latency_histogram.png'
)

# Print real-time performance summary
print("\n=== ECHTZEIT-PERFORMANCE ZUSAMMENFASSUNG ===")
print(f"ASTRA: {astra_mean_lat:.2f} ms Latenz, {astra_throughput:.1f} Samples/s Durchsatz")
print(f"GTSRB: {gtsrb_mean_lat:.2f} ms Latenz, {gtsrb_throughput:.1f} Samples/s Durchsatz")

# Create performance plots
print("\nErstelle Leistungsdiagramme...")
monitor.create_performance_plots(r'C:\Users\timau\Desktop\systemleistung.png')

# Total execution time
total_time = time.time() - start_time
print(f"\nGesamte Ausführungszeit: {total_time/60:.2f} Minuten")
print("Gespeicherte Dateien:")
print("- systemleistung.png")
print("- astra_confusion_matrix.png, gtsrb_confusion_matrix.png") 
print("- astra_latency_histogram.png, astra_latency_histogram_mit_statistiken.png")
print("- gtsrb_latency_histogram.png, gtsrb_latency_histogram_mit_statistiken.png")