import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import pandas as pd
import numpy as np
import time
import psutil
import GPUtil
from PIL import Image, ImageDraw
import os
import cv2
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings('ignore')

# Performance monitoring (adapted from GTSRB script)
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

# Data transformations for localization
train_transform = transforms.Compose([
    transforms.Resize((800, 1360)),  # Keep original aspect ratio
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    transforms.Resize((800, 1360)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Custom Dataset class for GTSDB
class GTSDBDataset(Dataset):
    def __init__(self, root_dir, gt_file, transform=None, max_objects=10):
        self.root_dir = root_dir
        self.transform = transform
        self.max_objects = max_objects
        
        # Parse ground truth file
        self.annotations = self._parse_gt_file(gt_file)
        
    def _parse_gt_file(self, gt_file):
        annotations = {}
        
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) == 6:
                    filename = parts[0].replace('.ppm', '.png')
                    x1, y1, x2, y2, class_id = map(int, parts[1:])
                    
                    if filename not in annotations:
                        annotations[filename] = []
                    
                    # Normalize coordinates to 0-1 range (based on original 1360x800)
                    x1_norm = x1 / 1360.0
                    y1_norm = y1 / 800.0
                    x2_norm = x2 / 1360.0
                    y2_norm = y2 / 800.0
                    
                    annotations[filename].append({
                        'x1': x1_norm, 'y1': y1_norm, 'x2': x2_norm, 'y2': y2_norm,
                        'class_id': class_id
                    })
        
        return annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        filename = list(self.annotations.keys())[idx]
        img_path = os.path.join(self.root_dir, filename)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        objects = self.annotations[filename]
        
        # Create target tensor: [num_objects, 5] -> [x1, y1, x2, y2, objectness]
        target = torch.zeros(self.max_objects, 5)
        
        for i, obj in enumerate(objects[:self.max_objects]):
            target[i, 0] = obj['x1']
            target[i, 1] = obj['y1'] 
            target[i, 2] = obj['x2']
            target[i, 3] = obj['y2']
            target[i, 4] = 1.0  # objectness (object present)
        
        if self.transform:
            image = self.transform(image)
            
        return image, target, filename

# Localization CNN Model
class LocalizationCNN(nn.Module):
    def __init__(self, max_objects=10):
        super(LocalizationCNN, self).__init__()
        self.max_objects = max_objects
        
        # Backbone feature extractor (similar to GTSRB style)
        self.backbone = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second block  
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fifth block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, max_objects * 5)  # 5 values per object: x1,y1,x2,y2,objectness
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Detection predictions
        detections = self.detection_head(features)
        detections = detections.view(-1, self.max_objects, 5)
        
        # Apply sigmoid to coordinates and objectness
        detections[:, :, :4] = torch.sigmoid(detections[:, :, :4])  # coordinates 0-1
        detections[:, :, 4] = torch.sigmoid(detections[:, :, 4])    # objectness 0-1
        
        return detections

# Custom loss function for localization
class LocalizationLoss(nn.Module):
    def __init__(self, coord_weight=5.0, obj_weight=1.0, noobj_weight=0.5):
        super(LocalizationLoss, self).__init__()
        self.coord_weight = coord_weight
        self.obj_weight = obj_weight
        self.noobj_weight = noobj_weight
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        
        # Separate coordinates and objectness
        pred_coords = predictions[:, :, :4]
        pred_obj = predictions[:, :, 4]
        
        target_coords = targets[:, :, :4]
        target_obj = targets[:, :, 4]
        
        # Object mask (where objects exist)
        obj_mask = target_obj > 0.5
        noobj_mask = target_obj <= 0.5
        
        # Coordinate loss (only for existing objects)
        coord_loss = 0
        if obj_mask.sum() > 0:
            coord_loss = self.mse(pred_coords[obj_mask], target_coords[obj_mask]).sum()
            coord_loss = coord_loss / batch_size
        
        # Objectness loss
        obj_loss = self.mse(pred_obj[obj_mask], target_obj[obj_mask]).sum() if obj_mask.sum() > 0 else 0
        noobj_loss = self.mse(pred_obj[noobj_mask], target_obj[noobj_mask]).sum() if noobj_mask.sum() > 0 else 0
        
        obj_loss = obj_loss / batch_size
        noobj_loss = noobj_loss / batch_size
        
        # Total loss
        total_loss = (self.coord_weight * coord_loss + 
                     self.obj_weight * obj_loss + 
                     self.noobj_weight * noobj_loss)
        
        return total_loss

# Training function
def train_localization_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, monitor=None):
    criterion = LocalizationLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Log performance every 5 epochs
        if monitor and epoch % 5 == 0:
            monitor.log_performance()
            
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
    return best_val_loss

# IoU calculation
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    if x2_min <= x1_max or y2_min <= y1_max:
        return 0.0
    
    intersection = (x2_min - x1_max) * (y2_min - y1_max)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

# Evaluation function for localization
def evaluate_localization_model(model, test_loader, iou_threshold=0.5, conf_threshold=0.5):
    model.eval()
    
    total_detections = 0
    total_ground_truths = 0
    true_positives = 0
    
    detection_results = []
    
    with torch.no_grad():
        for data, targets, filenames in test_loader:
            data = data.to(device)
            predictions = model(data)
            
            for i in range(data.size(0)):
                pred = predictions[i].cpu().numpy()
                target = targets[i].cpu().numpy()
                filename = filenames[i]
                
                # Filter predictions by confidence
                pred_boxes = []
                for j in range(pred.shape[0]):
                    if pred[j, 4] > conf_threshold:  # objectness threshold
                        pred_boxes.append(pred[j, :4])
                
                # Get ground truth boxes
                gt_boxes = []
                for j in range(target.shape[0]):
                    if target[j, 4] > 0.5:  # object exists
                        gt_boxes.append(target[j, :4])
                
                total_detections += len(pred_boxes)
                total_ground_truths += len(gt_boxes)
                
                # Calculate matches using IoU
                matched_gt = set()
                for pred_box in pred_boxes:
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if gt_idx not in matched_gt:
                            iou = calculate_iou(pred_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx
                    
                    if best_iou >= iou_threshold:
                        true_positives += 1
                        matched_gt.add(best_gt_idx)
                
                detection_results.append({
                    'filename': filename,
                    'pred_boxes': pred_boxes,
                    'gt_boxes': gt_boxes,
                    'pred_raw': pred
                })
    
    # Calculate metrics
    precision = true_positives / total_detections if total_detections > 0 else 0
    recall = true_positives / total_ground_truths if total_ground_truths > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score, detection_results

# Visualization function
def visualize_detections(detection_results, save_dir, max_images=10):
    """Create matplotlib visualizations of detection results"""
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, result in enumerate(detection_results[:max_images]):
        filename = result['filename']
        pred_boxes = result['pred_boxes']
        gt_boxes = result['gt_boxes']
        
        # Load original image
        img_path = os.path.join(r'C:\Users\timau\Desktop\Datensätze\GTSDB\Test', filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(r'C:\Users\timau\Desktop\Datensätze\GTSDB\Train', filename)
        
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
            img_width, img_height = image.size
            
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            ax.imshow(image)
            
            # Draw ground truth boxes (green)
            for gt_box in gt_boxes:
                x1 = gt_box[0] * img_width
                y1 = gt_box[1] * img_height
                x2 = gt_box[2] * img_width
                y2 = gt_box[3] * img_height
                
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=3, edgecolor='green', 
                                       facecolor='none', label='Ground Truth')
                ax.add_patch(rect)
            
            # Draw predicted boxes (red)
            for pred_box in pred_boxes:
                x1 = pred_box[0] * img_width
                y1 = pred_box[1] * img_height
                x2 = pred_box[2] * img_width
                y2 = pred_box[3] * img_height
                
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor='red', 
                                       facecolor='none', linestyle='--', 
                                       label='Vorhersage')
                ax.add_patch(rect)
            
            ax.set_title(f'Verkehrsschilder-Lokalisation - {filename}', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.axis('off')
            
            save_path = os.path.join(save_dir, f'detection_{idx:03d}_{filename}')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

# Extract 64x64 patches for classification
def extract_patches_for_classification(detection_results, patch_size=64, conf_threshold=0.5):
    """Extract 64x64 patches from detected regions for GTSRB classification"""
    patches = []
    patch_info = []
    
    for result in detection_results:
        filename = result['filename']
        pred_raw = result['pred_raw']
        
        # Load original image
        img_path = os.path.join(r'C:\Users\timau\Desktop\Datensätze\GTSDB\Test', filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(r'C:\Users\timau\Desktop\Datensätze\GTSDB\Train', filename)
        
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
            img_width, img_height = image.size
            
            # Extract patches from confident detections
            for i in range(pred_raw.shape[0]):
                if pred_raw[i, 4] > conf_threshold:  # confidence threshold
                    x1 = max(0, int(pred_raw[i, 0] * img_width))
                    y1 = max(0, int(pred_raw[i, 1] * img_height))
                    x2 = min(img_width, int(pred_raw[i, 2] * img_width))
                    y2 = min(img_height, int(pred_raw[i, 3] * img_height))
                    
                    # Extract and resize patch
                    patch = image.crop((x1, y1, x2, y2))
                    patch = patch.resize((patch_size, patch_size), Image.LANCZOS)
                    
                    patches.append(patch)
                    patch_info.append({
                        'filename': filename,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': pred_raw[i, 4]
                    })
    
    return patches, patch_info

# Real-time evaluation function (adapted from GTSRB script)
def realtime_evaluation_detailed(model, test_loader, model_name, warmup_batches=10):
    model.eval()
    
    # Warmup phase
    print(f"Warmup für {model_name}...")
    with torch.no_grad():
        for i, (data, _, _) in enumerate(test_loader):
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
    max_samples = 500  # Limit for reasonable test duration
    
    with torch.no_grad():
        for batch_data, _, _ in test_loader:
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

# Latency histogram function (adapted from GTSRB script)
def create_latency_histogram(latencies, model_name, save_path_with_stats, save_path_without_stats):
    mean_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    std_latency = np.std(latencies)
    
    # Separate latencies <= 50ms and > 50ms (adjusted for localization)
    latencies_filtered = [lat for lat in latencies if lat <= 50.0]
    latencies_over_50ms = [lat for lat in latencies if lat > 50.0]
    
    print(f"{model_name}: {len(latencies_over_50ms)} von {len(latencies)} Samples über 50ms ({len(latencies_over_50ms)/len(latencies)*100:.1f}%)")
    
    # Create bins: 0-50ms in regular intervals, plus one bin for >50ms
    regular_bins = np.linspace(0, 50, 40)  # 39 bins from 0-50ms
    
    # Histogram WITH statistics
    plt.figure(figsize=(12, 8))
    
    # Create histogram for ≤50ms data
    n, bins, patches = plt.hist(latencies_filtered, bins=regular_bins, alpha=0.7, 
                               color='steelblue', edgecolor='black')
    
    # Add the >50ms bin manually
    if len(latencies_over_50ms) > 0:
        bin_width = regular_bins[1] - regular_bins[0]
        plt.bar(50.0, len(latencies_over_50ms), width=bin_width, 
               alpha=0.7, color='red', edgecolor='black', label=f'>50ms (n={len(latencies_over_50ms)})')
    
    # Set y-axis to log scale
    plt.yscale('log')
    
    # Add vertical lines for mean and median
    if mean_latency <= 50.0:
        plt.axvline(mean_latency, color='red', linestyle='--', linewidth=2, 
                   label=f'Mittelwert: {mean_latency:.2f} ms')
    if median_latency <= 50.0:
        plt.axvline(median_latency, color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {median_latency:.2f} ms')
    
    plt.title(f'Latenz-Verteilung - {model_name} (mit Statistiken)', fontsize=16, fontweight='bold')
    plt.xlabel('Latenz pro Bild (ms)', fontweight='bold')
    plt.ylabel('Häufigkeit (log scale)', fontweight='bold')
    
    plt.xlim(0, 52)
    xticks = list(np.arange(0, 55, 10))
    xtick_labels = [f'{x:.0f}' if x <= 50.0 else '>50' for x in xticks]
    xtick_labels[-1] = '>50'
    plt.xticks(xticks, xtick_labels)
    
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path_without_stats, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mean_latency

# Classification with GTSRB model
class GTSRBClassifier:
    def __init__(self, model_path, device):
        self.device = device
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def _load_model(self, model_path):
        # Load the GTSRB model architecture (same as original script)
        class TrafficSignCNN(nn.Module):
            def __init__(self, num_classes=43):
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
        
        model = TrafficSignCNN(num_classes=43).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def classify_patches(self, patches):
        """Classify extracted patches"""
        if not patches:
            return []
        
        classifications = []
        
        with torch.no_grad():
            for patch in patches:
                # Transform patch
                patch_tensor = self.transform(patch).unsqueeze(0).to(self.device)
                
                # Get prediction
                output = self.model(patch_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                classifications.append({
                    'class': predicted_class.item(),
                    'confidence': confidence.item()
                })
        
        return classifications

# Main execution
print("=== GTSDB VERKEHRSSCHILDER-LOKALISATION ===")

# Load datasets
print("Lade GTSDB-Datensätze...")

# GTSDB Training dataset
gtsdb_train_dataset = GTSDBDataset(
    root_dir=r'C:\Users\timau\Desktop\Datensätze\GTSDB\Train',
    gt_file=r'C:\Users\timau\Desktop\Datensätze\GTSDB\gt-train.txt',
    transform=train_transform,
    max_objects=10
)

# GTSDB Test dataset
gtsdb_test_dataset = GTSDBDataset(
    root_dir=r'C:\Users\timau\Desktop\Datensätze\GTSDB\Test', 
    gt_file=r'C:\Users\timau\Desktop\Datensätze\GTSDB\gt-test.txt',
    transform=test_transform,
    max_objects=10
)

# Dataset information
print("\n=== DATENSATZ INFORMATIONEN ===")
print(f"GTSDB Training: {len(gtsdb_train_dataset)} Bilder")
print(f"GTSDB Test: {len(gtsdb_test_dataset)} Bilder")

# Train/Validation split
train_size = int(0.8 * len(gtsdb_train_dataset))
val_size = len(gtsdb_train_dataset) - train_size
gtsdb_train_split, gtsdb_val_split = random_split(gtsdb_train_dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(gtsdb_train_split, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(gtsdb_val_split, batch_size=8, shuffle=False, num_workers=2)
test_loader = DataLoader(gtsdb_test_dataset, batch_size=4, shuffle=False, num_workers=2)

print(f"Training Batches: {len(train_loader)}")
print(f"Validation Batches: {len(val_loader)}")
print(f"Test Batches: {len(test_loader)}")

# Initialize localization model
print("\n=== MODELL INITIALISIERUNG ===")
localization_model = LocalizationCNN(max_objects=10).to(device)

# Count parameters
total_params = sum(p.numel() for p in localization_model.parameters())
trainable_params = sum(p.numel() for p in localization_model.parameters() if p.requires_grad)
print(f"Lokalisations-Modell Parameter: {total_params:,}")
print(f"Trainierbare Parameter: {trainable_params:,}")

# Train localization model
print("\n=== LOKALISATIONS-MODELL TRAINING ===")
best_val_loss = train_localization_model(
    localization_model, train_loader, val_loader, 
    num_epochs=100, lr=0.001, monitor=monitor
)

# Save localization model
model_save_path = r'C:\Users\timau\Desktop\gtsdb_localization_model.pth'
torch.save(localization_model.state_dict(), model_save_path)
print(f"Lokalisations-Modell gespeichert: {model_save_path}")

# Evaluate localization model
print("\n=== LOKALISATIONS-EVALUATION ===")
precision, recall, f1_score, detection_results = evaluate_localization_model(
    localization_model, test_loader, iou_threshold=0.5, conf_threshold=0.3
)

print(f"Lokalisations-Performance:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

# Create detection visualizations
print("\nErstelle Lokalisations-Visualisierungen...")
vis_save_dir = r'C:\Users\timau\Desktop\detection_visualizations'
visualize_detections(detection_results, vis_save_dir, max_images=10)

# Extract patches for classification
print("\n=== PATCH-EXTRAKTION FÜR KLASSIFIKATION ===")
patches, patch_info = extract_patches_for_classification(
    detection_results, patch_size=64, conf_threshold=0.3
)
print(f"Extrahierte Patches: {len(patches)}")

# Load GTSRB classifier and classify patches
print("\n=== SCHILDKLASSIFIZIERUNG ===")
try:
    gtsrb_classifier = GTSRBClassifier(
        model_path=r'C:\Users\timau\Desktop\gtsrb_model.pth',
        device=device
    )
    
    classifications = gtsrb_classifier.classify_patches(patches)
    
    # Classification results
    print("Klassifizierungsergebnisse:")
    for i, (patch_info_item, classification) in enumerate(zip(patch_info, classifications)):
        print(f"Patch {i+1}: Klasse {classification['class']}, "
              f"Konfidenz: {classification['confidence']:.3f}, "
              f"Datei: {patch_info_item['filename']}")
    
    # Classification statistics
    if classifications:
        class_counts = {}
        confidences = []
        
        for cls in classifications:
            class_id = cls['class']
            confidence = cls['confidence']
            
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            confidences.append(confidence)
        
        print(f"\nKlassifizierungs-Statistiken:")
        print(f"Durchschnittliche Konfidenz: {np.mean(confidences):.3f}")
        print(f"Erkannte Klassen: {sorted(class_counts.keys())}")
        print(f"Häufigste Klasse: {max(class_counts, key=class_counts.get)} "
              f"({class_counts[max(class_counts, key=class_counts.get)]} mal)")

except Exception as e:
    print(f"Fehler beim Laden des GTSRB-Modells: {e}")
    print("Überspringe Klassifizierung...")

# Real-time performance evaluation
print("\n=== ECHTZEIT-PERFORMANCE EVALUATION ===")

# Localization model latency
loc_latencies, loc_throughput = realtime_evaluation_detailed(
    localization_model, test_loader, "GTSDB-Lokalisation"
)

# Create latency histograms
print("Erstelle Latenz-Histogramme...")
loc_mean_lat = create_latency_histogram(
    loc_latencies, "GTSDB-Lokalisation",
    r'C:\Users\timau\Desktop\gtsdb_localization_latency_histogram_mit_statistiken.png',
    r'C:\Users\timau\Desktop\gtsdb_localization_latency_histogram.png'
)

# Combined system evaluation (localization + classification)
print("\n=== KOMBINIERTE SYSTEM-EVALUATION ===")

if 'gtsrb_classifier' in locals():
    print("Messe End-to-End Latenz (Lokalisation + Klassifikation)...")
    
    combined_latencies = []
    total_samples = 0
    max_samples = 100
    
    with torch.no_grad():
        for batch_data, _, filenames in test_loader:
            for single_image, filename in zip(batch_data, filenames):
                if total_samples >= max_samples:
                    break
                
                # Single image processing
                single_image = single_image.unsqueeze(0).to(device)
                
                # Measure combined time
                start_time = time.time()
                
                # Localization
                predictions = localization_model(single_image)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                
                # Extract patches (simulation)
                pred = predictions[0].cpu().numpy()
                patch_count = 0
                for i in range(pred.shape[0]):
                    if pred[i, 4] > 0.3:  # confidence threshold
                        patch_count += 1
                
                # Simulate classification time (based on patch count)
                if patch_count > 0:
                    dummy_patches = [Image.new('RGB', (64, 64))] * patch_count
                    _ = gtsrb_classifier.classify_patches(dummy_patches[:3])  # max 3 patches
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                # Calculate combined latency
                latency_ms = (end_time - start_time) * 1000
                combined_latencies.append(latency_ms)
                
                total_samples += 1
            
            if total_samples >= max_samples:
                break
    
    # Create combined latency histogram
    combined_mean_lat = create_latency_histogram(
        combined_latencies, "End-to-End (Lokalisation + Klassifikation)",
        r'C:\Users\timau\Desktop\combined_system_latency_histogram_mit_statistiken.png',
        r'C:\Users\timau\Desktop\combined_system_latency_histogram.png'
    )
    
    print(f"End-to-End Performance: {combined_mean_lat:.2f} ms mittlere Latenz")

# Create performance plots
print("\nErstelle Systemleistungs-Diagramme...")
monitor.create_performance_plots(r'C:\Users\timau\Desktop\gtsdb_systemleistung.png')

# Performance summary
print("\n=== PERFORMANCE ZUSAMMENFASSUNG ===")
print(f"Lokalisations-Modell: {loc_mean_lat:.2f} ms Latenz, {loc_throughput:.1f} Bilder/s")
if 'combined_mean_lat' in locals():
    print(f"End-to-End System: {combined_mean_lat:.2f} ms Latenz")

# Final results summary
print(f"\n=== ENDERGEBNISSE ===")
print(f"Lokalisations-Performance:")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall: {recall:.4f}")
print(f"  - F1-Score: {f1_score:.4f}")
print(f"  - Latenz: {loc_mean_lat:.2f} ms")

if 'classifications' in locals() and classifications:
    print(f"Klassifizierung:")
    print(f"  - {len(patches)} Patches extrahiert")
    print(f"  - Durchschnittliche Konfidenz: {np.mean([c['confidence'] for c in classifications]):.3f}")

# Total execution time
total_time = time.time() - start_time
print(f"\nGesamte Ausführungszeit: {total_time/60:.2f} Minuten")

print("\nGespeicherte Dateien:")
print("- gtsdb_localization_model.pth")
print("- gtsdb_systemleistung.png") 
print("- gtsdb_localization_latency_histogram.png")
print("- gtsdb_localization_latency_histogram_mit_statistiken.png")
if 'combined_mean_lat' in locals():
    print("- combined_system_latency_histogram.png")
    print("- combined_system_latency_histogram_mit_statistiken.png")
print("- detection_visualizations/ (Ordner mit Visualisierungen)")