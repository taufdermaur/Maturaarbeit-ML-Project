import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from PIL import Image
import os
import cv2
import shutil
import random
import warnings
import yaml
from pathlib import Path
import subprocess
import sys
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from collections import Counter
import psutil
import threading
import GPUtil

warnings.filterwarnings('ignore')

# Start timer and setup
start_time = time.time()
print("="*80)
print("=== GTSDB PURE YOLO TRAINING & EVALUATION (43 CLASSES) ===")
print("="*80)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# CUDA Setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Install required packages
print("Installing required packages...")
try:
    import psutil
    import GPUtil
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'psutil', 'GPUtil'], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import psutil
    import GPUtil
subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics'], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics'], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
from ultralytics import YOLO
print("Ultralytics loaded")

# Resource monitoring class
class ResourceMonitor:
    def __init__(self):
        self.monitoring = False
        self.data = {
            'timestamps': [],
            'cpu_percent': [],
            'ram_percent': [],
            'gpu_percent': []
        }
    
    def start_monitoring(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_resources(self):
        while self.monitoring:
            try:
                # CPU and RAM
                cpu_percent = psutil.cpu_percent(interval=None)
                ram_percent = psutil.virtual_memory().percent
                
                # GPU
                gpu_percent = 0
                
                if torch.cuda.is_available():
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            gpu_percent = gpu.load * 100
                    except:
                        # No fallback for GPU usage - keep at 0
                        pass
                
                # Store data
                self.data['timestamps'].append(time.time())
                self.data['cpu_percent'].append(cpu_percent)
                self.data['ram_percent'].append(ram_percent)
                self.data['gpu_percent'].append(gpu_percent)
                
                time.sleep(2)  # Sample every 2 seconds
                
            except Exception:
                time.sleep(2)
    
    def get_stats(self):
        if not self.data['cpu_percent']:
            return None
        
        return {
            'cpu_avg': np.mean(self.data['cpu_percent']),
            'cpu_max': np.max(self.data['cpu_percent']),
            'ram_avg': np.mean(self.data['ram_percent']),
            'ram_max': np.max(self.data['ram_percent']),
            'gpu_avg': np.mean(self.data['gpu_percent']),
            'gpu_max': np.max(self.data['gpu_percent']),
            'samples': len(self.data['cpu_percent'])
        }

# Initialize resource monitor
resource_monitor = ResourceMonitor()

# Enhanced GTSDB Parser - Preserves all 43 classes
class GTSDBParser:
    def __init__(self, annotation_file):
        self.annotations = {}
        self.filename_mapping = {}
        self.class_counts = Counter()
        
        if not os.path.exists(annotation_file):
            print(f"Annotation file not found: {annotation_file}")
            return
            
        with open(annotation_file, 'r') as f:
            for line_num, line in enumerate(f):
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split(';')
                    if len(parts) >= 6:
                        filename = parts[0]
                        try:
                            x1, y1, x2, y2 = map(int, parts[1:5])
                            class_id = int(parts[5])  # Keep original class ID (0-42)
                            
                            # Track class distribution
                            self.class_counts[class_id] += 1
                            
                            base_name = os.path.splitext(filename)[0]
                            self.filename_mapping[filename] = filename
                            self.filename_mapping[base_name + '.png'] = filename
                            self.filename_mapping[base_name + '.jpg'] = filename
                            
                            if filename not in self.annotations:
                                self.annotations[filename] = []
                            self.annotations[filename].append({
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'class_id': class_id
                            })
                        except ValueError:
                            continue
    
    def get_annotations(self, filename):
        if filename in self.annotations:
            return self.annotations[filename]
        
        if filename in self.filename_mapping:
            original_name = self.filename_mapping[filename]
            return self.annotations.get(original_name, [])
        
        base_name = os.path.splitext(filename)[0]
        for ext in ['.ppm', '.png', '.jpg', '.jpeg']:
            test_name = base_name + ext
            if test_name in self.annotations:
                return self.annotations[test_name]
        
        return []

def convert_to_yolo(annotations, img_width, img_height):
    yolo_annotations = []
    for ann in annotations:
        x1, y1, x2, y2 = ann['x1'], ann['y1'], ann['x2'], ann['y2']
        
        if x1 >= x2 or y1 >= y2:
            continue
        if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
            continue
            
        x_center = (x1 + x2) / 2.0 / img_width
        y_center = (y1 + y2) / 2.0 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        yolo_annotations.append({
            'class_id': ann['class_id'],  # Preserve original class ID
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height
        })
    return yolo_annotations

def process_split(file_list, annotations, source_path, split_name, yolo_dataset_path):
    count = 0
    processed_boxes = 0
    class_distribution = Counter()
    
    for filename in file_list:
        if filename in annotations:
            actual_filename = filename.replace('.ppm', '.png')
            src = os.path.join(source_path, actual_filename)
            
            if not os.path.exists(src):
                src = os.path.join(source_path, filename)
                if not os.path.exists(src):
                    continue
                    
            dst = os.path.join(yolo_dataset_path, 'images', split_name, actual_filename)
            shutil.copy2(src, dst)
            
            try:
                img = Image.open(src)
                img_width, img_height = img.size
                yolo_anns = convert_to_yolo(annotations[filename], img_width, img_height)
                
                if len(yolo_anns) > 0:
                    label_file = actual_filename.replace('.png', '.txt').replace('.ppm', '.txt').replace('.jpg', '.txt')
                    label_path = os.path.join(yolo_dataset_path, 'labels', split_name, label_file)
                    with open(label_path, 'w') as f:
                        for ann in yolo_anns:
                            f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")
                            class_distribution[ann['class_id']] += 1
                    count += 1
                    processed_boxes += len(yolo_anns)
            except Exception:
                continue
    return count, processed_boxes, class_distribution

# Dataset Preparation
print("\n" + "="*80)
print("=== DATASET PREPARATION ===")
print("="*80)

base_path = r'C:\Users\timau\Desktop\Datensaetze\GTSDB'
yolo_dataset_path = r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results\yolo_dataset_43_classes'

# Parse annotations
train_ann_file = os.path.join(base_path, 'Train', 'gt-train.txt')
test_ann_file = os.path.join(base_path, 'gt-test.txt')

train_parser = GTSDBParser(train_ann_file)
test_parser = GTSDBParser(test_ann_file)

if len(train_parser.annotations) == 0:
    print("ERROR: No training annotations found!")
    sys.exit(1)

# Print class distribution
print("Class distribution in training data:")
for class_id, count in sorted(train_parser.class_counts.items()):
    print(f"  Class {class_id:2d}: {count:3d} samples")

print(f"Total classes found: {len(train_parser.class_counts)}")
print(f"Total training annotations: {sum(train_parser.class_counts.values())}")

# Create YOLO structure
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(yolo_dataset_path, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(yolo_dataset_path, 'labels', split), exist_ok=True)

# Split training data
train_files = list(train_parser.annotations.keys())
random.shuffle(train_files)
split_idx = int(0.8 * len(train_files))
train_split = train_files[:split_idx]
val_split = train_files[split_idx:]

# Process data
print("Processing datasets...")
train_count, train_boxes, train_class_dist = process_split(train_split, train_parser.annotations, 
                                                          os.path.join(base_path, 'Train'), 'train', yolo_dataset_path)
val_count, val_boxes, val_class_dist = process_split(val_split, train_parser.annotations, 
                                                     os.path.join(base_path, 'Train'), 'val', yolo_dataset_path)
test_count, test_boxes, test_class_dist = process_split(list(test_parser.annotations.keys()), test_parser.annotations, 
                                                       os.path.join(base_path, 'Test'), 'test', yolo_dataset_path)

print(f"Processed - Train: {train_count} images ({train_boxes} boxes)")
print(f"          - Val: {val_count} images ({val_boxes} boxes)")
print(f"          - Test: {test_count} images ({test_boxes} boxes)")

if train_count == 0:
    print("ERROR: No training images processed!")
    sys.exit(1)

# Create class names (0-42)
class_names = [f"class_{i}" for i in range(43)]

# Create YAML config for 43 classes
config_path = os.path.join(yolo_dataset_path, 'dataset.yaml')
with open(config_path, 'w') as f:
    yaml.dump({
        'path': yolo_dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 43,  # 43 classes instead of 1
        'names': class_names
    }, f)

print(f"Created YAML config with {43} classes")

# YOLO Training
print("\n" + "="*80)
print("=== YOLO TRAINING (43 CLASSES) ===")
print("="*80)

os.environ['CUDA_VISIBLE_DEVICES'] = '0' if torch.cuda.is_available() else ''
model = YOLO('yolov8n.pt')

if hasattr(model.model, 'parameters'):
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"YOLO model parameters: {total_params:,}")

print("\nTraining started...")
print(f"Configuration:")
print(f"  - Epochs: 300")
print(f"  - Batch Size: 16")
print(f"  - Image Size: 640")
print(f"  - Classes: 43")
print(f"  - Device: {device}")
print(f"  - Data Augmentation: Enabled")
print(f"  - Resource Monitoring: Enabled")

# Start resource monitoring
print("Starting resource monitoring...")
resource_monitor.start_monitoring()

training_start = time.time()

results = model.train(
    data=config_path,
    epochs=300,  # Increased for better convergence with more classes
    imgsz=640,
    batch=16,    # Reduced batch size for small dataset
    device=device,
    project=r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results',
    name='yolo_gtsdb_43classes',
    verbose=False,
    plots=False,
    workers=0,
    amp=True,
    cache=True,
    multi_scale=False,
    optimizer='AdamW',
    cos_lr=True,
    warmup_epochs=3.0,
    warmup_bias_lr=0.1,
    warmup_momentum=0.8,
    cls=1.0,  # Classification loss weight
    box=7.5,  # Box regression loss weight
    dfl=1.5,  # Distribution focal loss weight
    # Data Augmentation for traffic signs
    augment=True,
    degrees=15.0,      # Rotation for camera angles
    translate=0.1,     # Translation for vehicle movement
    scale=0.5,         # Scale for different distances
    shear=0.0,         # No shear for traffic signs
    perspective=0.0004, # Slight perspective distortion
    flipud=0.0,        # No vertical flip (preserves sign orientation)
    fliplr=0.0,        # No horizontal flip (preserves text/symbols)
    mosaic=1.0,        # Combine multiple images
    mixup=0.0,         # Disabled for classification tasks
    hsv_h=0.015,       # Hue variation for lighting conditions
    hsv_s=0.7,         # Saturation for weather conditions
    hsv_v=0.4,         # Brightness for day/night variation
    copy_paste=0.0     # Disabled for traffic signs
)

training_time = time.time() - training_start

# Find trained weights
trained_weights = os.path.join(r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results', 'yolo_gtsdb_43classes', 'weights', 'best.pt')
if not os.path.exists(trained_weights):
    # Try to find the most recent training folder
    project_dir = r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results'
    training_dirs = [d for d in os.listdir(project_dir) if d.startswith('yolo_gtsdb_43classes')]
    if training_dirs:
        latest_dir = sorted(training_dirs)[-1]
        trained_weights = os.path.join(project_dir, latest_dir, 'weights', 'best.pt')
    else:
        trained_weights = 'yolov8n.pt'

print(f"Training completed in {training_time/60:.1f} minutes")

# Stop resource monitoring and get stats  
resource_monitor.stop_monitoring()
training_resource_stats = resource_monitor.get_stats()

# Load trained model for evaluation
model = YOLO(trained_weights)
if hasattr(model.model, 'to'):
    model.model.to(device)

# ===========================================================================================
# PURE YOLO EVALUATION (43 CLASSES)
# ===========================================================================================
print("\n" + "="*80)
print("=== PURE YOLO EVALUATION (43 CLASSES) ===")
print("="*80)

test_images_path = os.path.join(yolo_dataset_path, 'images', 'test')
test_files = [f for f in os.listdir(test_images_path) if f.endswith(('.ppm', '.jpg', '.png'))]

print(f"Evaluating YOLO on {len(test_files)} test images...")

# Results storage
evaluation_results = {
    'inference_times': [],
    'all_predictions': [],
    'all_ground_truth': [],
    'per_class_metrics': {},
    'detection_confidences': [],
    'classification_confidences': [],
    'detailed_predictions': []
}

detected_count = 0
gt_count = 0
files_with_gt = 0

# Per-class tracking
y_true_all = []
y_pred_all = []
y_true_binary = []  # Has any detection
y_pred_binary = []  # Has any ground truth

evaluation_start = time.time()

# Warmup
print("Warming up YOLO model...")
for i, filename in enumerate(test_files[:min(10, len(test_files))]):
    img_path = os.path.join(test_images_path, filename)
    try:
        _ = model(img_path, conf=0.25, verbose=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    except Exception:
        continue

max_samples = min(1000, len(test_files))
print(f"Processing {max_samples} test images...")

for idx, filename in enumerate(test_files[:max_samples]):
    img_path = os.path.join(test_images_path, filename)
    
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    img_height, img_width = img.shape[:2]
    
    # YOLO inference with timing
    start_time_inference = time.time()
    try:
        results_list = model(img_path, conf=0.25, verbose=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_time = (time.time() - start_time_inference) * 1000
    except Exception:
        continue
    
    evaluation_results['inference_times'].append(inference_time)
    
    # Ground truth processing
    gt_boxes = test_parser.get_annotations(filename)
    if len(gt_boxes) > 0:
        files_with_gt += 1
        gt_count += len(gt_boxes)
        
        # Binary: image has ground truth
        y_true_binary.append(1)
        
        # Multi-class: collect all GT classes in this image
        gt_classes_in_image = [box['class_id'] for box in gt_boxes]
        
    else:
        y_true_binary.append(0)
        gt_classes_in_image = []
    
    # Process YOLO predictions
    pred_classes_in_image = []
    
    for result in results_list:
        if result.boxes is not None:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                pred_classes_in_image.append(cls)
                
                evaluation_results['detection_confidences'].append(float(conf))
                evaluation_results['classification_confidences'].append(float(conf))  # Same for pure YOLO
                
                # Store detailed prediction
                evaluation_results['detailed_predictions'].append({
                    'filename': filename,
                    'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                    'confidence': float(conf),
                    'class_id': cls,
                    'class_name': class_names[cls],
                    'image_width': img_width,
                    'image_height': img_height
                })
                
                detected_count += 1
    
    # Binary classification: has detection
    y_pred_binary.append(1 if len(pred_classes_in_image) > 0 else 0)
    
    # For multi-class evaluation: match predictions to ground truth
    # Simple approach: count each class present in the image
    all_classes_in_image = set(gt_classes_in_image + pred_classes_in_image)
    
    for class_id in all_classes_in_image:
        gt_present = 1 if class_id in gt_classes_in_image else 0
        pred_present = 1 if class_id in pred_classes_in_image else 0
        
        y_true_all.append(gt_present)
        y_pred_all.append(pred_present)
    
    if (idx + 1) % 100 == 0:
        print(f"Evaluation progress: {idx + 1}/{max_samples} ({((idx + 1)/max_samples*100):.1f}%)")

evaluation_time = time.time() - evaluation_start

# Calculate metrics
print(f"Evaluation completed in {evaluation_time:.1f} seconds")

# Binary metrics (detection vs no detection)
binary_accuracy = accuracy_score(y_true_binary, y_pred_binary)
binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
    y_true_binary, y_pred_binary, average='binary', zero_division=0
)

# Multi-class metrics
if len(y_true_all) > 0:
    multiclass_accuracy = accuracy_score(y_true_all, y_pred_all)
    multiclass_precision, multiclass_recall, multiclass_f1, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average='weighted', zero_division=0
    )
else:
    multiclass_accuracy = 0.0
    multiclass_precision = 0.0
    multiclass_recall = 0.0
    multiclass_f1 = 0.0

# Class distribution in predictions
pred_class_distribution = Counter()
gt_class_distribution = Counter()

for pred in evaluation_results['detailed_predictions']:
    pred_class_distribution[pred['class_id']] += 1

for filename in test_files[:max_samples]:
    gt_boxes = test_parser.get_annotations(filename)
    for box in gt_boxes:
        gt_class_distribution[box['class_id']] += 1

# ===========================================================================================
# PERFORMANCE ANALYSIS + RESULTS
# ===========================================================================================
print("\n" + "="*80)
print("=== PERFORMANCE ANALYSIS + RESULTS ===")
print("="*80)

# Timing Performance
if len(evaluation_results['inference_times']) > 0:
    latency_times = evaluation_results['inference_times']
    
    mean_latency = np.mean(latency_times)
    std_latency = np.std(latency_times)
    min_latency = np.min(latency_times)
    max_latency = np.max(latency_times)
    yolo_fps = 1000/mean_latency
    
    print("YOLO PERFORMANCE:")
    print(f"  Mittlere Latenz:           {mean_latency:.2f} ms")
    print(f"  Standardabweichung:        {std_latency:.2f} ms")
    print(f"  Min Latenz:                {min_latency:.2f} ms")
    print(f"  Max Latenz:                {max_latency:.2f} ms")
    print(f"  Durchsatz:                 {yolo_fps:.1f} FPS")
    
    # Latency distribution
    fast_count = len([t for t in latency_times if t <= 20])
    medium_count = len([t for t in latency_times if 20 < t <= 50])
    slow_count = len([t for t in latency_times if t > 50])
    
    print(f"  Latenz-Verteilung:")
    print(f"    ≤20ms:                   {fast_count} ({fast_count/len(latency_times)*100:.1f}%)")
    print(f"    20-50ms:                 {medium_count} ({medium_count/len(latency_times)*100:.1f}%)")
    print(f"    >50ms:                   {slow_count} ({slow_count/len(latency_times)*100:.1f}%)")

# Confidence Analysis
if evaluation_results['detection_confidences']:
    confidences = evaluation_results['detection_confidences']
    
    mean_conf = np.mean(confidences)
    std_conf = np.std(confidences)
    min_conf = np.min(confidences)
    max_conf = np.max(confidences)
    
    print(f"\nCONFIDENCE ANALYSIS:")
    print(f"  Mittlere Konfidenz:        {mean_conf:.3f}")
    print(f"  Standardabweichung:        {std_conf:.3f}")
    print(f"  Min Konfidenz:             {min_conf:.3f}")
    print(f"  Max Konfidenz:             {max_conf:.3f}")
    
    # Confidence distribution
    high_conf = len([c for c in confidences if c >= 0.8])
    medium_conf = len([c for c in confidences if 0.5 <= c < 0.8])
    low_conf = len([c for c in confidences if c < 0.5])
    
    print(f"  Konfidenz-Verteilung:")
    print(f"    ≥0.8:                    {high_conf} ({high_conf/len(confidences)*100:.1f}%)")
    print(f"    0.5-0.8:                 {medium_conf} ({medium_conf/len(confidences)*100:.1f}%)")
    print(f"    <0.5:                    {low_conf} ({low_conf/len(confidences)*100:.1f}%)")

# Detection Results
print(f"\nDETECTION ERGEBNISSE:")
print(f"  Verarbeitete Testbilder:   {len(evaluation_results['inference_times'])}")
print(f"  Dateien mit GT-Daten:      {files_with_gt}")
print(f"  Detektierte Objekte:       {detected_count}")
print(f"  Ground Truth Objekte:      {gt_count}")
if gt_count > 0:
    detection_rate = detected_count / gt_count * 100
    print(f"  Detection Rate:            {detection_rate:.1f}%")

# Binary Classification Metrics (Detection vs No Detection)
print(f"\nBINARY DETECTION METRIKEN:")
print(f"  Accuracy:                  {binary_accuracy:.4f}")
print(f"  Precision:                 {binary_precision:.4f}")
print(f"  Recall:                    {binary_recall:.4f}")
print(f"  F1-Score:                  {binary_f1:.4f}")

# Multi-Class Classification Metrics
print(f"\nMULTI-CLASS METRIKEN:")
print(f"  Accuracy:                  {multiclass_accuracy:.4f}")
print(f"  Precision:                 {multiclass_precision:.4f}")
print(f"  Recall:                    {multiclass_recall:.4f}")
print(f"  F1-Score:                  {multiclass_f1:.4f}")

# Class Distribution Analysis
print(f"\nCLASS DISTRIBUTION ANALYSIS:")
print(f"  Unique classes in GT:      {len(gt_class_distribution)}")
print(f"  Unique classes predicted:  {len(pred_class_distribution)}")

print(f"\nTOP 10 PREDICTED CLASSES:")
for i, (class_id, count) in enumerate(pred_class_distribution.most_common(10), 1):
    percentage = count / detected_count * 100 if detected_count > 0 else 0
    print(f"  {i:2d}. Class {class_id:2d}: {count:4d} ({percentage:5.1f}%)")

print(f"\nTOP 10 GROUND TRUTH CLASSES:")
for i, (class_id, count) in enumerate(gt_class_distribution.most_common(10), 1):
    percentage = count / gt_count * 100 if gt_count > 0 else 0
    print(f"  {i:2d}. Class {class_id:2d}: {count:4d} ({percentage:5.1f}%)")

# ===========================================================================================
# TIME SUMMARY
# ===========================================================================================
print("\n" + "="*80)
print("=== TIME SUMMARY ===")
print("="*80)

total_time = time.time() - start_time

print(f"\nEXECUTION TIMES:")
print(f"  Training Time:             {training_time/60:.2f} minutes")
print(f"  Evaluation Time:           {evaluation_time:.2f} seconds")
print(f"  Total Runtime:             {total_time/60:.2f} minutes")

# Resource usage during training
if training_resource_stats:
    print(f"\nRESOURCE USAGE DURING TRAINING:")
    print(f"  CPU Usage:")
    print(f"    Average:                 {training_resource_stats['cpu_avg']:.1f}%")
    print(f"    Maximum:                 {training_resource_stats['cpu_max']:.1f}%")
    print(f"  RAM Usage:")
    print(f"    Average:                 {training_resource_stats['ram_avg']:.1f}%")
    print(f"    Maximum:                 {training_resource_stats['ram_max']:.1f}%")
    if torch.cuda.is_available():
        print(f"  GPU Usage:")
        print(f"    Average:                 {training_resource_stats['gpu_avg']:.1f}%")
        print(f"    Maximum:                 {training_resource_stats['gpu_max']:.1f}%")
    print(f"  Monitoring Samples:        {training_resource_stats['samples']}")

print(f"\nTHROUGHPUT:")
if len(evaluation_results['inference_times']) > 0:
    print(f"  YOLO Inference:            {1000/np.mean(evaluation_results['inference_times']):.1f} FPS")

# ===========================================================================================
# MODEL AND DATASET STATISTICS
# ===========================================================================================
print("\n" + "="*80)
print("=== MODEL AND DATASET STATISTICS ===")
print("="*80)

print(f"MODEL PARAMETERS:")
print(f"  YOLO Parameters:           {total_params:,}")

print(f"\nMODEL CONFIGURATION:")
print(f"  YOLO Version:              YOLOv8n")
print(f"  Input Size:                640x640")
print(f"  Number of Classes:         43")
print(f"  Device:                    {device}")
print(f"  CUDA Available:            {torch.cuda.is_available()}")

print(f"\nDATASET STATISTICS:")
print(f"  Training Images:           {train_count}")
print(f"  Validation Images:         {val_count}")
print(f"  Test Images:               {test_count}")
print(f"  Training Boxes:            {train_boxes}")
print(f"  Validation Boxes:          {val_boxes}")
print(f"  Test Boxes:                {test_boxes}")
print(f"  Total Images:              {train_count + val_count + test_count}")
print(f"  Total Boxes:               {train_boxes + val_boxes + test_boxes}")

print(f"\nDATASET QUALITY:")
if (train_count + val_count + test_count) > 0:
    avg_boxes_per_image = (train_boxes + val_boxes + test_boxes)/(train_count + val_count + test_count)
    print(f"  Avg Boxes per Image:       {avg_boxes_per_image:.2f}")
print(f"  Test Coverage GT:          {files_with_gt}/{test_count} ({files_with_gt/test_count*100:.1f}%)")

# ===========================================================================================
# SAVE RESULTS
# ===========================================================================================
print(f"\nSaving results...")

# Define file paths
metadata_file = r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results\gtsdb_pure_yolo_metadata.pkl'
predictions_txt_file = r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results\gtsdb_pure_yolo_predictions.txt'

# Create results directory if it doesn't exist
os.makedirs(r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results', exist_ok=True)

# Save comprehensive results
metadata = {
    'trained_weights': trained_weights,
    'yolo_dataset_path': yolo_dataset_path,
    'config_path': config_path,
    'test_parser': test_parser,
    'start_time_script': start_time,
    'device': str(device),
    'evaluation_results': evaluation_results,
    'cuda_available': torch.cuda.is_available(),
    'files_with_gt': files_with_gt,
    'binary_accuracy': binary_accuracy,
    'binary_precision': binary_precision,
    'binary_recall': binary_recall,
    'binary_f1': binary_f1,
    'multiclass_accuracy': multiclass_accuracy,
    'multiclass_precision': multiclass_precision,
    'multiclass_recall': multiclass_recall,
    'multiclass_f1': multiclass_f1,
    'training_time': training_time,
    'evaluation_time': evaluation_time,
    'class_distribution_pred': dict(pred_class_distribution),
    'class_distribution_gt': dict(gt_class_distribution),
    'total_params': total_params,
    'training_resource_stats': training_resource_stats  # Added resource stats
}

with open(metadata_file, 'wb') as f:
    pickle.dump(metadata, f)

# Save predictions to TXT file
with open(predictions_txt_file, 'w') as f:
    f.write("# GTSDB Pure YOLO Predictions (43 Classes)\n")
    f.write("# Format: filename,x1,y1,x2,y2,confidence,class_id,class_name,image_width,image_height\n")
    
    for pred in evaluation_results['detailed_predictions']:
        f.write(f"{pred['filename']},{pred['x1']},{pred['y1']},{pred['x2']},{pred['y2']},{pred['confidence']:.3f},"
                f"{pred['class_id']},{pred['class_name']},{pred['image_width']},{pred['image_height']}\n")

# ===========================================================================================
# FINAL SUMMARY
# ===========================================================================================
print("\n" + "="*80)
print("=== FINAL SUMMARY ===")
print("="*80)

print(f"Training and evaluation completed successfully.")
print(f"Total runtime: {total_time/60:.2f} minutes")

print(f"\nCORE METRICS:")
if len(evaluation_results['inference_times']) > 0:
    print(f"  YOLO Durchsatz:            {1000/np.mean(evaluation_results['inference_times']):.1f} FPS")
    print(f"  Mittlere Latenz:           {np.mean(evaluation_results['inference_times']):.2f} ms")
    if evaluation_results['detection_confidences']:
        print(f"  Detection Konfidenz:       {np.mean(evaluation_results['detection_confidences']):.3f}")

print(f"\nDETECTION PERFORMANCE:")
print(f"  Binary Accuracy:           {binary_accuracy:.4f}")
print(f"  Binary Precision:          {binary_precision:.4f}")
print(f"  Binary Recall:             {binary_recall:.4f}")
print(f"  Binary F1-Score:           {binary_f1:.4f}")

print(f"\nCLASSIFICATION PERFORMANCE:")
print(f"  Multi-class Accuracy:      {multiclass_accuracy:.4f}")
print(f"  Multi-class Precision:     {multiclass_precision:.4f}")
print(f"  Multi-class Recall:        {multiclass_recall:.4f}")
print(f"  Multi-class F1-Score:      {multiclass_f1:.4f}")

print(f"\nSAVED FILES:")
print(f"  - {trained_weights}")
print(f"  - {metadata_file}")
print(f"  - {predictions_txt_file}")
print(f"  - {config_path}")

print(f"\n" + "="*80)
print("=== GTSDB PURE YOLO TRAINING & EVALUATION COMPLETED ===")
print("="*80)