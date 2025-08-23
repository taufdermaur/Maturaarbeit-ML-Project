import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
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

# GTSDB Dataset class for binary classification (sign/no sign)
class GTSDBDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.samples = []
        
        # Read annotations for positive samples (with signs)
        annotations = []
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(';')
                    if len(parts) >= 6:
                        filename = parts[0]
                        x1, y1, x2, y2 = map(int, parts[1:5])
                        
                        img_path = os.path.join(root_dir, filename)
                        if os.path.exists(img_path):
                            annotations.append({
                                'filename': filename,
                                'bbox': [x1, y1, x2, y2]
                            })
        
        # Create positive samples (class 1: sign exists)
        for ann in annotations:
            img_path = os.path.join(root_dir, ann['filename'])
            image = Image.open(img_path).convert('RGB')
            img_width, img_height = image.size
            
            x1, y1, x2, y2 = ann['bbox']
            
            # Expand bounding box slightly for better sign capture
            padding = 20
            x1_exp = max(0, x1 - padding)
            y1_exp = max(0, y1 - padding)
            x2_exp = min(img_width, x2 + padding)
            y2_exp = min(img_height, y2 + padding)
            
            # Crop sign region for positive sample
            sign_crop = image.crop((x1_exp, y1_exp, x2_exp, y2_exp))
            
            self.samples.append({
                'image': sign_crop,
                'label': 1,  # Class 1: sign
                'original_filename': ann['filename']
            })
        
        # Create negative samples (class 0: no sign)
        for ann in annotations:
            img_path = os.path.join(root_dir, ann['filename'])
            image = Image.open(img_path).convert('RGB')
            img_width, img_height = image.size
            
            x1, y1, x2, y2 = ann['bbox']
            
            # Create multiple negative samples per image
            neg_samples_per_image = 2
            for _ in range(neg_samples_per_image):
                attempts = 0
                while attempts < 20:
                    # Random crop size between 32 and 128 pixels
                    crop_size = np.random.randint(32, 128)
                    
                    # Random position
                    max_x = max(1, img_width - crop_size)
                    max_y = max(1, img_height - crop_size)
                    crop_x = np.random.randint(0, max_x)
                    crop_y = np.random.randint(0, max_y)
                    crop_x2 = crop_x + crop_size
                    crop_y2 = crop_y + crop_size
                    
                    # Check if crop overlaps significantly with sign bounding box
                    overlap_x = max(0, min(crop_x2, x2) - max(crop_x, x1))
                    overlap_y = max(0, min(crop_y2, y2) - max(crop_y, y1))
                    overlap_area = overlap_x * overlap_y
                    crop_area = crop_size * crop_size
                    
                    # If overlap is less than 10% of crop area, it's a good negative sample
                    if overlap_area / crop_area < 0.1:
                        negative_crop = image.crop((crop_x, crop_y, crop_x2, crop_y2))
                        
                        self.samples.append({
                            'image': negative_crop,
                            'label': 0,  # Class 0: no sign
                            'original_filename': ann['filename']
                        })
                        break
                    
                    attempts += 1
        
        # Shuffle samples
        np.random.shuffle(self.samples)
        
        # Count classes
        positive_count = sum(1 for s in self.samples if s['label'] == 1)
        negative_count = sum(1 for s in self.samples if s['label'] == 0)
        
        print(f"GTSDB Dataset loaded: {len(self.samples)} samples")
        print(f"Positive samples (sign): {positive_count}")
        print(f"Negative samples (no sign): {negative_count}")
        print(f"Class balance: {positive_count/(positive_count+negative_count)*100:.1f}% positive")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image']
        label = sample['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

# Binary Classification CNN Model
class TrafficSignBinaryClassifier(nn.Module):
    def __init__(self):
        super(TrafficSignBinaryClassifier, self).__init__()
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
            nn.Linear(512, 2)  # 2 classes: 0=no sign, 1=sign
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Classification CNN Model (same as in script 1)
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
def train_binary_classifier(model, train_loader, val_loader, num_epochs=25, lr=0.001, monitor=None):
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
def evaluate_binary_classifier(model, test_loader):
    model.eval()
    all_predictions = []
    all_targets = []
    all_confidences = []
    class_confidences = {0: [], 1: []}
    
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
            for pred, conf, true_label in zip(predicted.cpu().numpy(), 
                                            confidences.cpu().numpy(), 
                                            target.cpu().numpy()):
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
    
    plt.figure(figsize=(8, 6))
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
                    fontsize=14)
    
    # Set labels with class numbers only
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    
    plt.ylabel('Wahre Klasse', fontweight='bold')
    plt.xlabel('Vorhergesagte Klasse', fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Function to create latency histogram
def create_latency_histogram(latencies, model_name, save_path_with_stats, save_path_without_stats):
    mean_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    std_latency = np.std(latencies)
    
    # Histogram WITH statistics
    plt.figure(figsize=(12, 8))
    plt.hist(latencies, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add vertical lines for mean and median
    plt.axvline(mean_latency, color='red', linestyle='--', linewidth=2, label=f'Mittelwert: {mean_latency:.2f} ms')
    plt.axvline(median_latency, color='green', linestyle='--', linewidth=2, label=f'Median: {median_latency:.2f} ms')
    
    plt.title(f'Latenz-Verteilung - {model_name} (mit Statistiken)', fontsize=16, fontweight='bold')
    plt.xlabel('Latenz pro Bild (ms)', fontweight='bold')
    plt.ylabel('Häufigkeit', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Statistiken:\nMittelwert: {mean_latency:.2f} ms\nMedian: {median_latency:.2f} ms\nStd.-Abw.: {std_latency:.2f} ms\nMin: {min(latencies):.2f} ms\nMax: {max(latencies):.2f} ms'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path_with_stats, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Histogram WITHOUT statistics
    plt.figure(figsize=(12, 8))
    plt.hist(latencies, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    
    plt.title(f'Latenz-Verteilung - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Latenz pro Bild (ms)', fontweight='bold')
    plt.ylabel('Häufigkeit', fontweight='bold')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path_without_stats, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mean_latency

# Real-time evaluation function
def realtime_evaluation_binary(binary_model, classifier_model, test_loader, model_name, warmup_batches=10):
    binary_model.eval()
    classifier_model.eval()
    
    # Warmup phase
    print(f"Warmup für {model_name}...")
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= warmup_batches:
                break
            data = data.to(device)
            _ = binary_model(data)
            _ = classifier_model(data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Realistic single-image latency measurement
    print(f"Messe realistische Einzelbild-Latenz für {model_name}...")
    per_image_latencies = []
    total_samples = 0
    total_time = 0
    max_samples = 1000
    
    with torch.no_grad():
        for batch_data, _ in test_loader:
            # Process each image individually for realistic latency
            for single_image in batch_data:
                if total_samples >= max_samples:
                    break
                
                # Single image processing (batch_size = 1)
                single_image = single_image.unsqueeze(0).to(device)
                
                # Measure time for combined pipeline
                start_time = time.time()
                _ = binary_model(single_image)  # Detection
                _ = classifier_model(single_image)  # Classification
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
    throughput = total_samples / total_time if total_time > 0 else 0
    
    return per_image_latencies, throughput

# Load datasets
print("Lade Datensätze...")

gtsdb_train_dataset = GTSDBDataset(
    root_dir=r'C:\Users\timau\Desktop\Datensaetze\GTSDB\Train',
    annotation_file=r'C:\Users\timau\Desktop\Datensaetze\GTSDB\Train\gt-train.txt',
    transform=train_transform,
    is_train=True
)

gtsdb_test_dataset = GTSDBDataset(
    root_dir=r'C:\Users\timau\Desktop\Datensaetze\GTSDB\Test',
    annotation_file=r'C:\Users\timau\Desktop\Datensaetze\GTSDB\gt-test.txt',
    transform=test_transform,
    is_train=False
)

# Dataset information
print("\n=== DATENSATZ INFORMATIONEN ===")
print(f"GTSDB Training: {len(gtsdb_train_dataset)} Samples, 2 Klassen")
print(f"GTSDB Test: {len(gtsdb_test_dataset)} Samples, 2 Klassen")

# Train/Validation split
train_size = int(0.8 * len(gtsdb_train_dataset))
val_size = len(gtsdb_train_dataset) - train_size
gtsdb_train_split, gtsdb_val_split = random_split(gtsdb_train_dataset, [train_size, val_size])

gtsdb_train_loader = DataLoader(gtsdb_train_split, batch_size=16, shuffle=True)
gtsdb_val_loader = DataLoader(gtsdb_val_split, batch_size=16, shuffle=False)
gtsdb_test_loader = DataLoader(gtsdb_test_dataset, batch_size=16, shuffle=False)

# Train binary classification model
print("\n=== GTSDB MODELL TRAINING ===")
binary_model = TrafficSignBinaryClassifier().to(device)

# Count parameters
binary_total_params = sum(p.numel() for p in binary_model.parameters())
print(f"GTSDB Binary Modell Parameter: {binary_total_params:,}")

binary_best_acc = train_binary_classifier(binary_model, gtsdb_train_loader, gtsdb_val_loader, num_epochs=25, monitor=monitor)

# Save binary classification model
torch.save(binary_model.state_dict(), r'C:\Users\timau\Desktop\gtsdb_binary_model.pth')

# Function to classify detected signs with GTSRB model
def classify_detected_signs(binary_model, classifier_model, test_loader, threshold=0.5):
    """
    Use binary model to detect signs, then classify detected signs with GTSRB classifier
    """
    binary_model.eval()
    classifier_model.eval()
    
    detected_signs = []
    sign_classifications = []
    actual_labels = []  # For evaluation (we'll use a dummy approach)
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            
            # Step 1: Binary detection
            binary_output = binary_model(batch_data)
            binary_probs = torch.softmax(binary_output, dim=1)
            _, binary_pred = torch.max(binary_probs, 1)
            
            # Step 2: Classify only detected signs (binary_pred == 1)
            for i, (data, binary_prediction, original_label) in enumerate(zip(batch_data, binary_pred, batch_labels)):
                if binary_prediction == 1:  # Sign detected
                    # Classify the sign
                    single_image = data.unsqueeze(0)
                    class_output = classifier_model(single_image)
                    _, predicted_class = torch.max(class_output, 1)
                    
                    detected_signs.append(1)  # Sign was detected
                    sign_classifications.append(predicted_class.item())
                    # For evaluation, we'll use a random class (since we don't have ground truth GTSRB labels)
                    actual_labels.append(np.random.randint(0, 43))  # Dummy labels for demo
                else:
                    detected_signs.append(0)  # No sign detected
    
    return detected_signs, sign_classifications, actual_labels

# Evaluation function for sign classification pipeline
def evaluate_sign_classification_pipeline(binary_model, classifier_model, test_loader):
    """
    Evaluate the complete pipeline: detection + classification
    """
    binary_model.eval()
    classifier_model.eval()
    
    # Binary detection metrics
    binary_predictions = []
    binary_targets = []
    
    # Classification metrics (only for detected signs)
    classification_predictions = []
    classification_targets = []
    
    total_images = 0
    signs_detected = 0
    signs_classified = 0
    
    with torch.no_grad():
        for batch_data, batch_binary_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_binary_labels = batch_binary_labels.to(device)
            
            # Binary detection
            binary_output = binary_model(batch_data)
            _, binary_pred = torch.max(binary_output, 1)
            
            binary_predictions.extend(binary_pred.cpu().numpy())
            binary_targets.extend(batch_binary_labels.cpu().numpy())
            
            total_images += len(batch_data)
            
            # For each detected sign, perform classification
            for i, (single_data, binary_prediction, true_binary_label) in enumerate(zip(batch_data, binary_pred, batch_binary_labels)):
                if binary_prediction == 1:  # Sign detected
                    signs_detected += 1
                    
                    # Classify the detected sign
                    single_image = single_data.unsqueeze(0)
                    class_output = classifier_model(single_image)
                    _, predicted_class = torch.max(class_output, 1)
                    
                    classification_predictions.append(predicted_class.item())
                    # Generate dummy ground truth for classification (since we don't have real GTSRB labels)
                    classification_targets.append(np.random.randint(0, 43))
                    signs_classified += 1
    
    # Calculate binary detection metrics
    binary_accuracy = accuracy_score(binary_targets, binary_predictions)
    binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
        binary_targets, binary_predictions, average='weighted'
    )
    
    # Calculate classification metrics (only for detected signs)
    if len(classification_predictions) > 0:
        class_accuracy = accuracy_score(classification_targets, classification_predictions)
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            classification_targets, classification_predictions, average='weighted'
        )
    else:
        class_accuracy = class_precision = class_recall = class_f1 = 0.0
    
    # Pipeline statistics
    detection_rate = signs_detected / total_images if total_images > 0 else 0
    classification_rate = signs_classified / signs_detected if signs_detected > 0 else 0
    
    return {
        'binary_accuracy': binary_accuracy,
        'binary_precision': binary_precision, 
        'binary_recall': binary_recall,
        'binary_f1': binary_f1,
        'classification_accuracy': class_accuracy,
        'classification_precision': class_precision,
        'classification_recall': class_recall,
        'classification_f1': class_f1,
        'total_images': total_images,
        'signs_detected': signs_detected,
        'signs_classified': signs_classified,
        'detection_rate': detection_rate,
        'classification_rate': classification_rate
    }

# Load GTSRB classification model
print("\nLade GTSRB Klassifikationsmodell...")
gtsrb_num_classes = 43  # GTSRB has 43 classes
classification_model = TrafficSignCNN(gtsrb_num_classes).to(device)

# Try to load pre-trained GTSRB model
try:
    classification_model.load_state_dict(torch.load(r'C:\Users\timau\Desktop\gtsrb_model.pth'))
    print("GTSRB-Modell erfolgreich geladen")
except FileNotFoundError:
    print("GTSRB-Modell nicht gefunden. Verwende zufällig initialisiertes Modell.")

# Evaluation
print("\n=== MODELL EVALUATION ===")

binary_accuracy, binary_precision, binary_recall, binary_f1, binary_predictions, binary_targets, binary_confidences = evaluate_binary_classifier(
    binary_model, gtsdb_test_loader
)

# Results table
results_data = {
    'Model': ['GTSDB Binary Classification'],
    'Accuracy': [f'{binary_accuracy:.4f}'],
    'Precision': [f'{binary_precision:.4f}'],
    'Recall': [f'{binary_recall:.4f}'],
    'F1-Score': [f'{binary_f1:.4f}']
}

results_df = pd.DataFrame(results_data)
print("\nEvaluierungsergebnisse:")
print(results_df.to_string(index=False))

# Confidence Analysis
print("\n=== KONFIDENZ ANALYSE ===")
print(f"\nGTSDB Binary Classification - Durchschnittliche Konfidenz pro Klasse:")
binary_conf_sorted = sorted(binary_confidences.items())
for class_id, confidence in binary_conf_sorted:
    class_name = "Kein Schild" if class_id == 0 else "Schild"
    print(f"Klasse {class_id} ({class_name}): {confidence:.4f}")

# Pipeline Evaluation: Detection + Classification
print("\n=== PIPELINE EVALUATION: DETEKTION + KLASSIFIKATION ===")

pipeline_results = evaluate_sign_classification_pipeline(binary_model, classification_model, gtsdb_test_loader)

print(f"\nPipeline Statistiken:")
print(f"Gesamt Bilder: {pipeline_results['total_images']}")
print(f"Schilder detektiert: {pipeline_results['signs_detected']}")
print(f"Schilder klassifiziert: {pipeline_results['signs_classified']}")
print(f"Detektionsrate: {pipeline_results['detection_rate']:.2%}")
print(f"Klassifikationsrate: {pipeline_results['classification_rate']:.2%}")

print(f"\nDetektion (Binär) Metriken:")
print(f"Accuracy: {pipeline_results['binary_accuracy']:.4f}")
print(f"Precision: {pipeline_results['binary_precision']:.4f}")
print(f"Recall: {pipeline_results['binary_recall']:.4f}")
print(f"F1-Score: {pipeline_results['binary_f1']:.4f}")

print(f"\nKlassifikation (GTSRB) Metriken:")
print(f"Accuracy: {pipeline_results['classification_accuracy']:.4f}")
print(f"Precision: {pipeline_results['classification_precision']:.4f}")
print(f"Recall: {pipeline_results['classification_recall']:.4f}")
print(f"F1-Score: {pipeline_results['classification_f1']:.4f}")

# Note about dummy labels
print(f"\nHinweis: Klassifikations-Metriken verwenden Dummy-Labels für Demonstration,")
print(f"da GTSDB keine GTSRB-Klassifikations-Ground-Truth enthält.")

# Create visualizations
print("\n=== ERSTELLE VISUALISIERUNGEN ===")

# Confusion matrix
print("Erstelle Konfusionsmatrix...")
create_confusion_matrix(binary_targets, binary_predictions, 
                       2,  # Binary classification
                       "GTSDB Binary Classification", 
                       r'C:\Users\timau\Desktop\gtsdb_confusion_matrix.png')

# Real-time performance evaluation
print("\n=== ECHTZEIT-PERFORMANCE EVALUATION ===")

combined_latencies, combined_throughput = realtime_evaluation_binary(
    binary_model, classification_model, gtsdb_test_loader, "GTSDB Pipeline"
)

# Create latency histograms
print("Erstelle Latenz-Histogramme...")
combined_mean_lat = create_latency_histogram(
    combined_latencies, "GTSDB Pipeline (Detection + Classification)",
    r'C:\Users\timau\Desktop\gtsdb_latency_histogram_mit_statistiken.png',
    r'C:\Users\timau\Desktop\gtsdb_latency_histogram.png'
)

# Print real-time performance summary
print("\n=== ECHTZEIT-PERFORMANCE ZUSAMMENFASSUNG ===")
print(f"GTSDB Pipeline: {combined_mean_lat:.2f} ms Latenz, {combined_throughput:.1f} Samples/s Durchsatz")

# Create performance plots
print("\nErstelle Leistungsdiagramme...")
monitor.create_performance_plots(r'C:\Users\timau\Desktop\gtsdb_systemleistung.png')

# Total execution time
total_time = time.time() - start_time
print(f"\nGesamte Ausführungszeit: {total_time/60:.2f} Minuten")
print("Gespeicherte Dateien:")
print("- gtsdb_systemleistung.png")
print("- gtsdb_confusion_matrix.png") 
print("- gtsdb_latency_histogram.png, gtsdb_latency_histogram_mit_statistiken.png")