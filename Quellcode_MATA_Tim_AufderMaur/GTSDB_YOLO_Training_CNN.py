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
from collections import Counter
import threading

warnings.filterwarnings('ignore')

# Start timer and setup
start_time = time.time()
print("="*80)
print("=== GTSDB HYBRID PIPELINE: TRAINING SCRIPT ===")
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
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'psutil'], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import psutil

subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics'], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
from ultralytics import YOLO
print("Packages loaded")

# Resource monitoring class
class ResourceMonitor:
    def __init__(self):
        self.monitoring = False
        self.data = {
            'timestamps': [],
            'cpu_percent': [],
            'ram_percent': []
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
                cpu_percent = psutil.cpu_percent(interval=None)
                ram_percent = psutil.virtual_memory().percent
                
                self.data['timestamps'].append(time.time())
                self.data['cpu_percent'].append(cpu_percent)
                self.data['ram_percent'].append(ram_percent)
                
                time.sleep(2)
                
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
            'samples': len(self.data['cpu_percent'])
        }

# Initialize resource monitor
resource_monitor = ResourceMonitor()

# GTSDB Parser - for single class detection training
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
                            original_class_id = int(parts[5])
                            
                            self.class_counts[original_class_id] += 1
                            
                            base_name = os.path.splitext(filename)[0]
                            self.filename_mapping[filename] = filename
                            self.filename_mapping[base_name + '.png'] = filename
                            self.filename_mapping[base_name + '.jpg'] = filename
                            
                            if filename not in self.annotations:
                                self.annotations[filename] = []
                            self.annotations[filename].append({
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'original_class_id': original_class_id,
                                'yolo_class_id': 0  # All become class 0 for detection
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
            'class_id': 0,  # Single class for detection
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height,
            'original_class_id': ann['original_class_id']
        })
    return yolo_annotations

def process_split(file_list, annotations, source_path, split_name, yolo_dataset_path):
    count = 0
    processed_boxes = 0
    
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
                            f.write(f"0 {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")
                    count += 1
                    processed_boxes += len(yolo_anns)
            except Exception:
                continue
    return count, processed_boxes

# Dataset Preparation
print("\n" + "="*80)
print("=== DATASET PREPARATION ===")
print("="*80)

base_path = r'C:\Users\timau\Desktop\Datensaetze\GTSDB'
yolo_dataset_path = r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results\yolo_dataset_detection_only'

# Parse annotations
train_ann_file = os.path.join(base_path, 'Train', 'gt-train.txt')
test_ann_file = os.path.join(base_path, 'gt-test.txt')

train_parser = GTSDBParser(train_ann_file)
test_parser = GTSDBParser(test_ann_file)

if len(train_parser.annotations) == 0:
    print("ERROR: No training annotations found!")
    sys.exit(1)

# Print original class distribution
print("Original class distribution in training data:")
for class_id, count in sorted(train_parser.class_counts.items()):
    print(f"  Class {class_id:2d}: {count:3d} samples")
print(f"Total classes found: {len(train_parser.class_counts)}")
print(f"Total training annotations: {sum(train_parser.class_counts.values())}")
print("Converting all classes to single detection class for YOLO training...")

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
train_count, train_boxes = process_split(train_split, train_parser.annotations, 
                                        os.path.join(base_path, 'Train'), 'train', yolo_dataset_path)
val_count, val_boxes = process_split(val_split, train_parser.annotations, 
                                    os.path.join(base_path, 'Train'), 'val', yolo_dataset_path)
test_count, test_boxes = process_split(list(test_parser.annotations.keys()), test_parser.annotations, 
                                      os.path.join(base_path, 'Test'), 'test', yolo_dataset_path)

print(f"Processed - Train: {train_count} images ({train_boxes} boxes)")
print(f"          - Val: {val_count} images ({val_boxes} boxes)")
print(f"          - Test: {test_count} images ({test_boxes} boxes)")

if train_count == 0:
    print("ERROR: No training images processed!")
    sys.exit(1)

# Create YAML config for single class detection
config_path = os.path.join(yolo_dataset_path, 'dataset.yaml')
with open(config_path, 'w') as f:
    yaml.dump({
        'path': yolo_dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': ['traffic_sign']
    }, f)

print(f"Created YAML config for single-class detection")

# YOLO Training for Detection Only
print("\n" + "="*80)
print("=== YOLO TRAINING (DETECTION ONLY) ===")
print("="*80)

os.environ['CUDA_VISIBLE_DEVICES'] = '0' if torch.cuda.is_available() else ''
model = YOLO('yolov8n.pt')

if hasattr(model.model, 'parameters'):
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"YOLO model parameters: {total_params:,}")

print("\nTraining started...")
print(f"Configuration:")
print(f"  - Epochs: 300")
print(f"  - Batch Size: 32")
print(f"  - Image Size: 640")
print(f"  - Classes: 1 (detection only)")
print(f"  - Device: {device}")
print(f"  - Data Augmentation: Enabled")
print(f"  - Resource Monitoring: Enabled")

# Start resource monitoring
print("Starting resource monitoring...")
resource_monitor.start_monitoring()

training_start = time.time()

results = model.train(
    data=config_path,
    epochs=300,
    imgsz=640,
    batch=32,
    device=device,
    project=r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results',
    name='yolo_detection_only',
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
    cls=1.0,
    box=7.5,
    dfl=1.5,
    # Data Augmentation
    augment=True,
    degrees=15.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0004,
    flipud=0.0,
    fliplr=0.0,
    mosaic=1.0,
    mixup=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    copy_paste=0.0
)

training_time = time.time() - training_start

# Stop resource monitoring and get stats
resource_monitor.stop_monitoring()
training_resource_stats = resource_monitor.get_stats()

print(f"Training completed in {training_time/60:.1f} minutes")

# Find trained weights
trained_weights = os.path.join(r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results', 'yolo_detection_only', 'weights', 'best.pt')
if not os.path.exists(trained_weights):
    project_dir = r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results'
    if os.path.exists(project_dir):
        training_dirs = [d for d in os.listdir(project_dir) if d.startswith('yolo_detection_only')]
        if training_dirs:
            latest_dir = sorted(training_dirs)[-1]
            trained_weights = os.path.join(project_dir, latest_dir, 'weights', 'best.pt')
        else:
            trained_weights = 'yolov8n.pt'
    else:
        trained_weights = 'yolov8n.pt'

# ===========================================================================================
# TRAINING SUMMARY
# ===========================================================================================
print("\n" + "="*80)
print("=== TRAINING SUMMARY ===")
print("="*80)

total_time = time.time() - start_time

print(f"TRAINING STATISTICS:")
print(f"  Training Time:             {training_time/60:.2f} minutes")
print(f"  Total Runtime:             {total_time/60:.2f} minutes")
print(f"  Model Parameters:          {total_params:,}")

# Resource usage during training
if training_resource_stats:
    print(f"\nRESOURCE USAGE DURING TRAINING:")
    print(f"  CPU Usage:")
    print(f"    Average:                 {training_resource_stats['cpu_avg']:.1f}%")
    print(f"    Maximum:                 {training_resource_stats['cpu_max']:.1f}%")
    print(f"  RAM Usage:")
    print(f"    Average:                 {training_resource_stats['ram_avg']:.1f}%")
    print(f"    Maximum:                 {training_resource_stats['ram_max']:.1f}%")
    print(f"  Monitoring Samples:        {training_resource_stats['samples']}")

print(f"\nDATASET STATISTICS:")
print(f"  Training Images:           {train_count}")
print(f"  Validation Images:         {val_count}")
print(f"  Test Images:               {test_count}")
print(f"  Training Boxes:            {train_boxes}")
print(f"  Validation Boxes:          {val_boxes}")
print(f"  Test Boxes:                {test_boxes}")

# Save training metadata
print(f"\nSaving training results...")

os.makedirs(r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results', exist_ok=True)

training_metadata = {
    'trained_weights': trained_weights,
    'yolo_dataset_path': yolo_dataset_path,
    'config_path': config_path,
    'train_parser': train_parser,
    'test_parser': test_parser,
    'start_time': start_time,
    'device': str(device),
    'training_time': training_time,
    'total_params': total_params,
    'training_resource_stats': training_resource_stats,
    'dataset_stats': {
        'train_count': train_count,
        'val_count': val_count,
        'test_count': test_count,
        'train_boxes': train_boxes,
        'val_boxes': val_boxes,
        'test_boxes': test_boxes
    },
    'class_distribution': dict(train_parser.class_counts),
    'cuda_available': torch.cuda.is_available()
}

training_metadata_file = r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results\training_metadata.pkl'
with open(training_metadata_file, 'wb') as f:
    pickle.dump(training_metadata, f)

print(f"\nSAVED FILES:")
print(f"  - {trained_weights}")
print(f"  - {training_metadata_file}")
print(f"  - {config_path}")
print(f"  - {yolo_dataset_path} (complete dataset)")

print(f"\n" + "="*80)
print("=== TRAINING COMPLETED SUCCESSFULLY ===")
print("="*80)
print(f"Ready for evaluation. Use trained weights: {trained_weights}")
print("="*80)