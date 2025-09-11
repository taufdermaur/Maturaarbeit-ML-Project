import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import time
import psutil
import GPUtil
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Data transformations for YOLO
test_transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Custom Dataset class for GTSDB with YOLO format
class GTSDBYOLODataset(Dataset):
    def __init__(self, root_dir, gt_file, transform=None, img_size=640):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
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
                    
                    # Convert to YOLO format (center_x, center_y, width, height)
                    center_x = ((x1 + x2) / 2) / 1360.0
                    center_y = ((y1 + y2) / 2) / 800.0
                    width = (x2 - x1) / 1360.0
                    height = (y2 - y1) / 800.0
                    
                    annotations[filename].append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': width,
                        'height': height,
                        'class_id': 0  # Single class: traffic sign
                    })
        
        return annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        filename = list(self.annotations.keys())[idx]
        img_path = os.path.join(self.root_dir, filename)
        
        image = Image.open(img_path).convert('RGB')
        objects = self.annotations[filename]
        
        targets = []
        for obj in objects:
            targets.append([
                0,  # batch_id
                obj['class_id'],
                obj['center_x'],
                obj['center_y'],
                obj['width'],
                obj['height']
            ])
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(targets, dtype=torch.float32), filename

# YOLO Model Implementation
class YOLOv5s(nn.Module):
    def __init__(self, num_classes=1):
        super(YOLOv5s, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = 3
        self.num_outputs = 5 + num_classes
        
        # Backbone (simplified CSPDarknet53)
        self.backbone = nn.Sequential(
            self._make_conv_block(3, 32, 6, 2, 2),
            self._make_conv_block(32, 64, 3, 2, 1),
            self._make_csp_block(64, 64, 1),
            self._make_conv_block(64, 128, 3, 2, 1),
            self._make_csp_block(128, 128, 3),
            self._make_conv_block(128, 256, 3, 2, 1),
            self._make_csp_block(256, 256, 3),
            self._make_conv_block(256, 512, 3, 2, 1),
            self._make_csp_block(512, 512, 1),
            self._make_conv_block(512, 1024, 3, 2, 1),
            self._make_csp_block(1024, 1024, 1),
        )
        
        # Neck (simplified FPN + PAN)
        self.neck = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
        )
        
        # Detection heads for different scales
        self.head_large = self._make_detection_head(512, self.num_outputs)
        self.head_medium = self._make_detection_head(256, self.num_outputs)
        self.head_small = self._make_detection_head(128, self.num_outputs)
        
        # Upsampling for multi-scale detection
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Additional conv layers for medium and small scales
        self.conv_medium = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )
        
        self.conv_small = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
        )
        
    def _make_conv_block(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def _make_csp_block(self, channels, channels_out, num_blocks):
        return nn.Sequential(
            nn.Conv2d(channels, channels_out, 1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.SiLU(inplace=True),
            *[self._make_residual_block(channels_out) for _ in range(num_blocks)]
        )
    
    def _make_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels//2, 1, bias=False),
            nn.BatchNorm2d(channels//2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels//2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
    
    def _make_detection_head(self, in_channels, num_outputs):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels, self.num_anchors * num_outputs, 1)
        )
    
    def forward(self, x):
        # Backbone
        features = self.backbone(x)
        
        # Neck
        neck_features = self.neck(features)
        
        # Large scale detection (20x20)
        large_out = self.head_large(neck_features)
        
        # Medium scale detection (40x40)
        medium_features = self.upsample(neck_features)
        medium_features = self.conv_medium(medium_features)
        medium_out = self.head_medium(medium_features)
        
        # Small scale detection (80x80)
        small_features = self.upsample(medium_features)
        small_features = self.conv_small(small_features)
        small_out = self.head_small(small_features)
        
        return [large_out, medium_out, small_out]

# KORRIGIERTE Non-Maximum Suppression
def non_max_suppression(predictions, conf_threshold=0.25, iou_threshold=0.45):
    """Apply NMS to predictions - KORRIGIERT"""
    batch_detections = []
    
    for pred_idx, pred in enumerate(predictions):
        # pred shape: [batch_size, anchors*outputs, grid_h, grid_w]
        batch_size, channels, grid_h, grid_w = pred.shape
        num_anchors = 3
        num_outputs = channels // num_anchors
        
        # Reshape: [batch_size, anchors, outputs, grid_h, grid_w] -> [batch_size, anchors, grid_h, grid_w, outputs]
        pred = pred.view(batch_size, num_anchors, num_outputs, grid_h, grid_w).permute(0, 1, 3, 4, 2)
        
        for batch_idx in range(batch_size):
            detections = []
            
            # Get detections for this batch
            batch_pred = pred[batch_idx]  # [anchors, grid_h, grid_w, outputs]
            
            for anchor_idx in range(num_anchors):
                for i in range(grid_h):
                    for j in range(grid_w):
                        detection = batch_pred[anchor_idx, i, j]  # [outputs]
                        
                        if len(detection) >= 5:  # Ensure we have enough outputs
                            # Extract values with .item() to avoid tensor indexing issues
                            center_x = (j + torch.sigmoid(detection[0]).item()) / grid_w
                            center_y = (i + torch.sigmoid(detection[1]).item()) / grid_h
                            width = torch.exp(detection[2]).item() * 0.1
                            height = torch.exp(detection[3]).item() * 0.1
                            conf = torch.sigmoid(detection[4]).item()
                            
                            if conf > conf_threshold:
                                # Convert to x1, y1, x2, y2 format
                                x1 = center_x - width / 2
                                y1 = center_y - height / 2
                                x2 = center_x + width / 2
                                y2 = center_y + height / 2
                                
                                detections.append([x1, y1, x2, y2, conf, 0])  # class 0
            
            # Apply simplified NMS
            if detections:
                detections = np.array(detections)
                keep = simple_nms(detections[:, :4], detections[:, 4], iou_threshold)
                batch_detections.append(torch.tensor(detections[keep]))
            else:
                batch_detections.append(torch.empty(0, 6))
    
    return batch_detections

# Simplified NMS implementation
def simple_nms(boxes, scores, iou_threshold):
    """Simplified NMS implementation"""
    if len(boxes) == 0:
        return []
    
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]
        
        ious = []
        for box in remaining_boxes:
            iou = calculate_iou(current_box, box)
            ious.append(iou)
        
        indices = indices[1:][np.array(ious) < iou_threshold]
    
    return keep

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

# Calculate mAP at different IoU thresholds
def calculate_map(detection_results, iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
    """Calculate mean Average Precision (mAP) for different IoU thresholds"""
    aps = []
    
    for iou_thresh in iou_thresholds:
        total_detections = 0
        total_ground_truths = 0
        true_positives = 0
        
        for result in detection_results:
            pred_boxes = result['pred_boxes']
            gt_boxes = result['gt_boxes']
            
            total_detections += len(pred_boxes)
            total_ground_truths += len(gt_boxes)
            
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
                
                if best_iou >= iou_thresh:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
        
        precision = true_positives / total_detections if total_detections > 0 else 0
        recall = true_positives / total_ground_truths if total_ground_truths > 0 else 0
        ap = precision * recall if (precision + recall) > 0 else 0
        aps.append(ap)
    
    return np.mean(aps), aps

# Evaluation function for YOLO
def evaluate_yolo_model(model, test_loader, conf_threshold=0.25, iou_threshold=0.5):
    model.eval()
    
    total_detections = 0
    total_ground_truths = 0
    true_positives = 0
    detection_results = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            data = torch.stack([item[0] for item in batch_data]).to(device)
            targets = [item[1] for item in batch_data]
            filenames = [item[2] for item in batch_data]
            
            predictions = model(data)
            
            batch_detections = non_max_suppression(
                [pred.cpu() for pred in predictions], 
                conf_threshold=conf_threshold
            )
            
            for i in range(data.size(0)):
                detections = batch_detections[i] if i < len(batch_detections) else torch.empty(0, 6)
                target = targets[i] if i < len(targets) else []
                filename = filenames[i]
                
                pred_boxes = []
                if len(detections) > 0:
                    for detection in detections:
                        pred_boxes.append(detection[:4].tolist())
                
                gt_boxes = []
                for target_obj in target:
                    if len(target_obj) >= 6:
                        _, _, center_x, center_y, width, height = target_obj[:6]
                        x1 = center_x - width / 2
                        y1 = center_y - height / 2
                        x2 = center_x + width / 2
                        y2 = center_y + height / 2
                        gt_boxes.append([x1, y1, x2, y2])
                
                total_detections += len(pred_boxes)
                total_ground_truths += len(gt_boxes)
                
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
                    'detections': detections.numpy() if len(detections) > 0 else np.array([])
                })
    
    precision = true_positives / total_detections if total_detections > 0 else 0
    recall = true_positives / total_ground_truths if total_ground_truths > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score, detection_results

# Visualization function for YOLO
def visualize_yolo_detections(detection_results, save_dir, max_images=10):
    """Create matplotlib visualizations of YOLO detection results"""
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, result in enumerate(detection_results[:max_images]):
        filename = result['filename']
        pred_boxes = result['pred_boxes']
        gt_boxes = result['gt_boxes']
        
        img_path = os.path.join(r'C:\Users\timau\Desktop\Datensaetze\GTSDB\Test', filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(r'C:\Users\timau\Desktop\Datensaetze\GTSDB\Train', filename)
        
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
            img_width, img_height = image.size
            
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            ax.imshow(image)
            
            for gt_box in gt_boxes:
                x1 = gt_box[0] * img_width
                y1 = gt_box[1] * img_height
                x2 = gt_box[2] * img_width
                y2 = gt_box[3] * img_height
                
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=3, edgecolor='green', 
                                       facecolor='none', label='Ground Truth')
                ax.add_patch(rect)
            
            for pred_box in pred_boxes:
                x1 = pred_box[0] * img_width
                y1 = pred_box[1] * img_height
                x2 = pred_box[2] * img_width
                y2 = pred_box[3] * img_height
                
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor='red', 
                                       facecolor='none', linestyle='--', 
                                       label='YOLO Vorhersage')
                ax.add_patch(rect)
            
            ax.set_title(f'YOLO Verkehrsschilder-Erkennung - {filename}', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.axis('off')
            
            save_path = os.path.join(save_dir, f'yolo_detection_{idx:03d}_{filename}')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

# Performance evaluation function
def realtime_evaluation_yolo(model, test_loader, model_name, warmup_batches=10):
    model.eval()
    
    print(f"Warmup für {model_name}...")
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            if i >= warmup_batches:
                break
            data = torch.stack([item[0] for item in batch_data]).to(device)
            _ = model(data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    print(f"Messe realistische Einzelbild-Latenz für {model_name}...")
    per_image_latencies = []
    total_samples = 0
    total_time = 0
    max_samples = 500
    
    with torch.no_grad():
        for batch_data in test_loader:
            for single_item in batch_data:
                if total_samples >= max_samples:
                    break
                
                single_image = single_item[0].unsqueeze(0).to(device)
                
                start_time = time.time()
                predictions = model(single_image)
                _ = non_max_suppression([pred.cpu() for pred in predictions])
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                per_image_latencies.append(latency_ms)
                
                total_time += (end_time - start_time)
                total_samples += 1
            
            if total_samples >= max_samples:
                break
    
    throughput = total_samples / total_time if total_time > 0 else 0
    
    return per_image_latencies, throughput

# Create latency histogram
def create_yolo_latency_histogram(latencies, model_name, save_path):
    mean_latency = np.mean(latencies)
    
    threshold = 50.0
    latencies_filtered = [lat for lat in latencies if lat <= threshold]
    latencies_over_threshold = [lat for lat in latencies if lat > threshold]
    
    print(f"{model_name}: {len(latencies_over_threshold)} von {len(latencies)} Samples über {threshold}ms ({len(latencies_over_threshold)/len(latencies)*100:.1f}%)")
    
    regular_bins = np.linspace(0, threshold, 40)
    
    plt.figure(figsize=(12, 8))
    
    plt.hist(latencies_filtered, bins=regular_bins, alpha=0.7, 
            color='steelblue', edgecolor='black')
    
    if len(latencies_over_threshold) > 0:
        bin_width = regular_bins[1] - regular_bins[0]
        plt.bar(threshold, len(latencies_over_threshold), width=bin_width, 
               alpha=0.7, color='red', edgecolor='black')
    
    plt.yscale('log')
    
    if mean_latency <= threshold:
        plt.axvline(mean_latency, color='red', linestyle='--', linewidth=2, 
                   label=f'Mittelwert: {mean_latency:.2f} ms')
    
    plt.title(f'YOLO Latenz-Verteilung - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Latenz pro Bild (ms)', fontweight='bold')
    plt.ylabel('Häufigkeit (log scale)', fontweight='bold')
    
    plt.xlim(0, threshold + 5)
    xticks = list(np.arange(0, threshold + 10, 10))
    xtick_labels = [f'{x:.0f}' if x <= threshold else f'>{threshold:.0f}' for x in xticks]
    plt.xticks(xticks, xtick_labels)
    
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mean_latency

# MAIN EVALUATION
print("=== YOLO EVALUATION-ONLY SKRIPT ===")

# Load test dataset
print("Lade GTSDB Test-Dataset...")
gtsdb_test_dataset = GTSDBYOLODataset(
    root_dir=r'C:\Users\timau\Desktop\Datensaetze\GTSDB\Test', 
    gt_file=r'C:\Users\timau\Desktop\Datensaetze\GTSDB\gt-test.txt',
    transform=test_transform,
    img_size=640
)

test_loader = DataLoader(gtsdb_test_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=lambda x: x)
print(f"Test Dataset: {len(gtsdb_test_dataset)} Bilder")

# Load trained YOLO model
print("\n=== LADE TRAINIERTES YOLO MODELL ===")
yolo_model = YOLOv5s(num_classes=1).to(device)
yolo_model.load_state_dict(torch.load(r'C:\Users\timau\Desktop\gtsdb_yolo_model.pth', map_location=device))
yolo_model.eval()

total_params = sum(p.numel() for p in yolo_model.parameters())
print(f"YOLO Modell geladen: {total_params:,} Parameter")

# Evaluate YOLO model
print("\n=== YOLO EVALUATION ===")
precision, recall, f1_score, detection_results = evaluate_yolo_model(
    yolo_model, test_loader, conf_threshold=0.25, iou_threshold=0.5
)

# Calculate mAP
map_score, ap_scores = calculate_map(detection_results)

print(f"YOLO Performance:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
print(f"mAP@0.5:0.95: {map_score:.4f}")
print(f"AP@0.5: {ap_scores[0]:.4f}")
print(f"AP@0.75: {ap_scores[5]:.4f}")

# Create visualizations
print("\n=== ERSTELLE VISUALISIERUNGEN ===")
vis_save_dir = r'C:\Users\timau\Desktop\yolo_detection_visualizations'
visualize_yolo_detections(detection_results, vis_save_dir, max_images=10)

# Performance evaluation
print("\n=== PERFORMANCE EVALUATION ===")
yolo_latencies, yolo_throughput = realtime_evaluation_yolo(
    yolo_model, test_loader, "YOLO v5s"
)

yolo_mean_lat = create_yolo_latency_histogram(
    yolo_latencies, "YOLO v5s",
    r'C:\Users\timau\Desktop\gtsdb_yolo_latency_histogram.png'
)

# Final results
fps_30 = 1000 / 30
fps_60 = 1000 / 60

print(f"\n=== FINALE ERGEBNISSE ===")
print(f"YOLO Erkennungs-Performance:")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall: {recall:.4f}")
print(f"  - F1-Score: {f1_score:.4f}")
print(f"  - mAP@0.5:0.95: {map_score:.4f}")
print(f"  - Latenz: {yolo_mean_lat:.2f} ms")
print(f"  - Durchsatz: {yolo_throughput:.1f} Bilder/s")

print(f"\nEchtzeit-Bewertung:")
if yolo_mean_lat <= fps_30:
    print("✓ YOLO kann 30 FPS erreichen")
else:
    print("✗ YOLO zu langsam für 30 FPS")

if yolo_mean_lat <= fps_60:
    print("✓ YOLO kann 60 FPS erreichen")
else:
    print("✗ YOLO zu langsam für 60 FPS")

print("\nGespeicherte Dateien:")
print("- gtsdb_yolo_latency_histogram.png")
print("- yolo_detection_visualizations/ (Ordner)")

print("\n" + "="*50)
print("✅ YOLO EVALUATION ABGESCHLOSSEN")
print("="*50)