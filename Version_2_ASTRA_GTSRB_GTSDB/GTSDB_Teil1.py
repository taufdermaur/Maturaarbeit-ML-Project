import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import time
import psutil
from PIL import Image
import os
import json
import csv
import math
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importiere optional
try:
    from torchvision import transforms, models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except:
    GPUTIL_AVAILABLE = False

# Konstanten für Lokalisierung
IMAGE_WIDTH = 1360
IMAGE_HEIGHT = 800
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# Desktop Pfad für extrahierte Schilder
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
EXTRACTED_SIGNS_DIR = os.path.join(DESKTOP_PATH, "ExtractedTrafficSigns")

# Performance Monitoring
class PerformanceMonitor:
    def __init__(self):
        self.cpu_history = []
        self.memory_history = []
        self.gpu_memory_history = []
        self.timestamps = []
        
    def log_performance(self):
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        gpu_memory = 0
        if GPUTIL_AVAILABLE and torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            gpu_memory = gpu.memoryUtil * 100
        
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_percent)
        self.gpu_memory_history.append(gpu_memory)
        self.timestamps.append(time.time())
    
    def get_summary(self):
        def safe_mean(x): return sum(x) / len(x) if x else 0
        def safe_max(x): return max(x) if x else 0
        
        return {
            'avg_cpu': safe_mean(self.cpu_history),
            'avg_memory': safe_mean(self.memory_history),
            'avg_gpu_memory': safe_mean(self.gpu_memory_history),
            'max_cpu': safe_max(self.cpu_history),
            'max_memory': safe_max(self.memory_history),
            'max_gpu_memory': safe_max(self.gpu_memory_history),
        }

# ECA-Net Attention Block
class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        k = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)

# Smooth L1 Loss für Bounding Box Regression
class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
    
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        cond = diff < self.beta
        loss = torch.where(cond, 0.5 * diff ** 2 / self.beta, diff - 0.5 * self.beta)
        return loss.mean()

# IoU Loss für bessere Lokalisierung
class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
    
    def forward(self, pred_boxes, target_boxes):
        # pred_boxes und target_boxes: [batch_size, 4] (x1, y1, x2, y2)
        
        # Intersection
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Union
        area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area_target = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union = area_pred + area_target - intersection
        
        # IoU
        iou = intersection / (union + 1e-6)
        
        # IoU Loss (1 - IoU)
        return (1 - iou).mean()

# Dataset für reine Bounding Box Lokalisierung
class GTSDBLocalizationDataset(Dataset):
    """Dataset für reine Bounding Box Lokalisierung (ohne Klassifikation)"""
    
    def __init__(self, images_dir, labels_file, transform=None, is_train=True):
        self.images_dir = images_dir
        self.transform = transform
        self.is_train = is_train
        
        # Labels laden - nehme nur Bilder mit EINEM Verkehrsschild
        self.samples = []
        image_annotations = {}
        
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) == 6:
                    filename = parts[0].replace('.ppm', '.png')
                    x1, y1, x2, y2, class_id = map(int, parts[1:])
                    
                    if filename not in image_annotations:
                        image_annotations[filename] = []
                    
                    image_annotations[filename].append({
                        'bbox': [x1, y1, x2, y2],
                        'class_id': class_id
                    })
        
        # Filtere Bilder mit genau einem Objekt für einfachere Lokalisierung
        single_object_images = {k: v for k, v in image_annotations.items() if len(v) == 1}
        
        for filename, annotations in single_object_images.items():
            bbox = annotations[0]['bbox']
            class_id = annotations[0]['class_id']  # Für Referenz, aber nicht für Training
            self.samples.append({
                'filename': filename,
                'bbox': bbox,
                'class_id': class_id
            })
        
        print(f"Geladene Einzelobjekt-Bilder: {len(self.samples)}")
        if len(image_annotations) > len(single_object_images):
            print(f"Ignorierte Mehrobjekt-Bilder: {len(image_annotations) - len(single_object_images)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        filename = sample['filename']
        bbox = sample['bbox']
        class_id = sample['class_id']  # Für Referenz
        
        # Bild laden
        image_path = os.path.join(self.images_dir, filename)
        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size
        
        # Resize Bild
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Skaliere Bounding Box
        scale_x = IMAGE_WIDTH / original_width
        scale_y = IMAGE_HEIGHT / original_height
        
        x1, y1, x2, y2 = bbox
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y
        
        # Normalisiere auf [0, 1]
        bbox_normalized = torch.tensor([
            x1_scaled / IMAGE_WIDTH,
            y1_scaled / IMAGE_HEIGHT,
            x2_scaled / IMAGE_WIDTH,
            y2_scaled / IMAGE_HEIGHT
        ], dtype=torch.float32)
        
        # Transformationen
        if self.transform:
            image = self.transform(image)
        
        return image, bbox_normalized, filename

# Convolutional Block mit ECA Attention
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_eca=False):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)
        
        self.use_eca = use_eca
        if use_eca:
            self.eca = ECABlock(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        if self.use_eca:
            x = self.eca(x)
        
        return x

# Lokalisierungs-CNN mit vortrainiertem Backbone
class TrafficSignLocalizer(nn.Module):
    """CNN mit vortrainiertem ResNet Backbone für bessere Konvergenz"""
    
    def __init__(self):
        super(TrafficSignLocalizer, self).__init__()
        
        # Verwende vortrainiertes ResNet18 als Backbone
        if TORCHVISION_AVAILABLE:
            backbone = models.resnet18(pretrained=True)
            # Entferne die letzte FC-Schicht
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone_out_channels = 512
        else:
            # Fallback zu einfacherem Backbone
            self.backbone = nn.Sequential(
                ConvBlock(3, 64, 7, 2, 3),   # Größere Kernel für bessere Features
                nn.MaxPool2d(2, 2),
                ConvBlock(64, 128, 3, 1, 1, use_eca=True),
                nn.MaxPool2d(2, 2),
                ConvBlock(128, 256, 3, 1, 1, use_eca=True),
                nn.MaxPool2d(2, 2),
                ConvBlock(256, 512, 3, 1, 1, use_eca=True),
                nn.MaxPool2d(2, 2),
            )
            backbone_out_channels = 512
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Vereinfachter Regressor mit besserer Initialisierung
        self.bbox_regressor = nn.Sequential(
            nn.Linear(backbone_out_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 4)  # Direkte Ausgabe ohne komplexe Transformationen
        )
        
        # Spezielle Initialisierung für Bounding Box Regression
        self._init_bbox_head()
    
    def _init_bbox_head(self):
        """Initialisiere den Bounding Box Head mit sinnvollen Werten"""
        # Initialisiere letzte Schicht so, dass sie mittlere Bildkoordinaten produziert
        with torch.no_grad():
            # Setze Bias so, dass Standard-Output etwa [0.3, 0.3, 0.7, 0.7] ist
            self.bbox_regressor[-1].bias.data = torch.tensor([0.3, 0.3, 0.7, 0.7])
            # Kleine Gewichte für sanften Start
            self.bbox_regressor[-1].weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        # Feature Extraction
        features = self.backbone(x)
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        
        # Direkte Bounding Box Regression
        bbox_raw = self.bbox_regressor(flattened)
        
        # Sanfte Sigmoid-Aktivierung
        bbox_pred = torch.sigmoid(bbox_raw)
        
        # Stelle sicher, dass x2 > x1 und y2 > y1
        x1, y1, x2, y2 = bbox_pred.split(1, dim=1)
        
        # Enforze gültige Box-Geometrie mit korrekter torch.clamp Syntax
        x1 = torch.clamp(x1, 0.0, 0.8)  # x1 nicht zu weit rechts
        y1 = torch.clamp(y1, 0.0, 0.8)  # y1 nicht zu weit unten
        
        # Berechne minimale x2, y2 Werte und clamp korrekt
        min_x2 = x1 + 0.1  # x2 mindestens x1 + 0.1
        min_y2 = y1 + 0.1  # y2 mindestens y1 + 0.1
        
        x2 = torch.clamp(x2, 0.0, 1.0)  # x2 in gültigem Bereich
        y2 = torch.clamp(y2, 0.0, 1.0)  # y2 in gültigem Bereich
        
        # Stelle sicher, dass x2 >= x1 + 0.1 und y2 >= y1 + 0.1
        x2 = torch.max(x2, min_x2)
        y2 = torch.max(y2, min_y2)
        
        # Final clamp um sicherzustellen, dass alles in [0,1] bleibt
        x2 = torch.clamp(x2, 0.0, 1.0)
        y2 = torch.clamp(y2, 0.0, 1.0)
        
        bbox_pred = torch.cat([x1, y1, x2, y2], dim=1)
        
        return bbox_pred
    
    def get_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

# Reine Lokalisierungs-Loss
class LocalizationLoss(nn.Module):
    def __init__(self):
        super(LocalizationLoss, self).__init__()
        self.iou_loss = IoULoss()
        self.smooth_l1_loss = SmoothL1Loss(beta=1.0)
    
    def forward(self, bbox_pred, bbox_target):
        # Adaptive Loss-Strategie
        with torch.no_grad():
            avg_iou = compute_iou(bbox_pred, bbox_target).mean()
        
        # Erst nur L1 Loss, dann IoU dazunehmen
        l1_loss = self.smooth_l1_loss(bbox_pred, bbox_target)
        
        if avg_iou < 0.1:
            bbox_loss = l1_loss  # Nur L1 Loss
        else:
            iou_loss = self.iou_loss(bbox_pred, bbox_target)
            bbox_loss = 0.7 * l1_loss + 0.3 * iou_loss  # Kombiniere beide
        
        return bbox_loss

def compute_iou(pred_boxes, target_boxes):
    """Berechnet IoU zwischen predicted und target boxes"""
    if len(pred_boxes.shape) == 1:
        pred_boxes = pred_boxes.unsqueeze(0)
        target_boxes = target_boxes.unsqueeze(0)
    
    # Intersection
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Union
    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_target = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union = area_pred + area_target - intersection
    
    # IoU
    iou = intersection / (union + 1e-6)
    return iou

def train_localizer(model, train_loader, val_loader, num_epochs=50, lr=0.01, monitor=None):
    """Training des reinen Lokalisierungs-Modells"""
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, verbose=True, min_lr=1e-6
    )
    
    criterion = LocalizationLoss()
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    best_val_iou = 0.0
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    
    for epoch in range(num_epochs):
        if monitor and epoch % 5 == 0:
            monitor.log_performance()
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_samples = 0
        
        for batch_idx, (data, bbox_targets, _) in enumerate(train_loader):
            data = data.to(model.device)
            bbox_targets = bbox_targets.to(model.device)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    bbox_pred = model(data)
                    loss = criterion(bbox_pred, bbox_targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                bbox_pred = model(data)
                loss = criterion(bbox_pred, bbox_targets)
                
                loss.backward()
                optimizer.step()
            
            # Metriken berechnen
            with torch.no_grad():
                batch_iou = compute_iou(bbox_pred, bbox_targets).mean()
                train_iou += batch_iou.item() * data.size(0)
                train_samples += data.size(0)
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, IoU: {batch_iou.item():.3f}')
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for data, bbox_targets, _ in val_loader:
                data = data.to(model.device)
                bbox_targets = bbox_targets.to(model.device)
                
                if scaler:
                    with autocast():
                        bbox_pred = model(data)
                        loss = criterion(bbox_pred, bbox_targets)
                else:
                    bbox_pred = model(data)
                    loss = criterion(bbox_pred, bbox_targets)
                
                # IoU berechnen
                batch_iou = compute_iou(bbox_pred, bbox_targets).mean()
                val_iou += batch_iou.item() * data.size(0)
                val_samples += data.size(0)
                
                val_loss += loss.item()
        
        # Durchschnitte berechnen
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_iou /= train_samples
        val_iou /= val_samples
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'Train IoU: {train_iou:.3f} | Val IoU: {val_iou:.3f}')
        
        # Learning Rate Scheduler
        scheduler.step(val_iou)
        
        # Bestes Modell speichern
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), 'best_traffic_sign_localizer.pth')
            print(f'Neues bestes Modell gespeichert (Val IoU: {val_iou:.3f})')
    
        print(f"Pred: x1={bbox_pred[0][0]:.3f}, y1={bbox_pred[0][1]:.3f}, x2={bbox_pred[0][2]:.3f}, y2={bbox_pred[0][3]:.3f}")
        print(f"Target: x1={bbox_targets[0][0]:.3f}, y1={bbox_targets[0][1]:.3f}, x2={bbox_targets[0][2]:.3f}, y2={bbox_targets[0][3]:.3f}")

        # Box-Größen prüfen
        pred_w = bbox_pred[0][2] - bbox_pred[0][0] 
        pred_h = bbox_pred[0][3] - bbox_pred[0][1]
        target_w = bbox_targets[0][2] - bbox_targets[0][0]
        target_h = bbox_targets[0][3] - bbox_targets[0][1]
        print(f"Pred box size: {pred_w:.3f} x {pred_h:.3f}")
        print(f"Target box size: {target_w:.3f} x {target_h:.3f}")

    return train_losses, val_losses, train_ious, val_ious

def extract_traffic_signs(model, test_loader, save_dir=EXTRACTED_SIGNS_DIR, confidence_threshold=0.3):
    """Extrahiert erkannte Verkehrsschilder und speichert sie auf Desktop"""
    
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n=== VERKEHRSSCHILDER EXTRAKTION ===")
    print(f"Speichere extrahierte Schilder in: {save_dir}")
    
    extracted_count = 0
    total_images = 0
    extraction_log = []
    
    with torch.no_grad():
        for batch_idx, (data, bbox_targets, filenames) in enumerate(test_loader):
            data = data.to(model.device)
            
            if torch.cuda.is_available():
                with autocast():
                    bbox_pred = model(data)
            else:
                bbox_pred = model(data)
            
            for i, filename in enumerate(filenames):
                total_images += 1
                
                # Berechne IoU als Confidence-Maß
                iou = compute_iou(bbox_pred[i:i+1], bbox_targets[i:i+1]).item()
                
                if iou >= confidence_threshold:
                    # Lade Originalbild
                    image_path = os.path.join(test_loader.dataset.images_dir, filename)
                    original_image = Image.open(image_path).convert('RGB')
                    
                    # Denormalisiere Bounding Box
                    pred_bbox = bbox_pred[i].cpu() * torch.tensor([IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT])
                    x1, y1, x2, y2 = pred_bbox.int().tolist()
                    
                    # Skaliere zurück zur Originalgröße
                    orig_w, orig_h = original_image.size
                    scale_x = orig_w / IMAGE_WIDTH
                    scale_y = orig_h / IMAGE_HEIGHT
                    
                    x1_orig = int(x1 * scale_x)
                    y1_orig = int(y1 * scale_y)
                    x2_orig = int(x2 * scale_x)
                    y2_orig = int(y2 * scale_y)
                    
                    # Clamp Koordinaten
                    x1_orig = max(0, min(x1_orig, orig_w))
                    y1_orig = max(0, min(y1_orig, orig_h))
                    x2_orig = max(0, min(x2_orig, orig_w))
                    y2_orig = max(0, min(y2_orig, orig_h))
                    
                    # Extrahiere Verkehrsschild
                    if x2_orig > x1_orig and y2_orig > y1_orig:
                        sign_crop = original_image.crop((x1_orig, y1_orig, x2_orig, y2_orig))
                        
                        # Speichere extrahiertes Schild
                        sign_filename = f"sign_{extracted_count:04d}_{filename.replace('.png', '')}_iou{iou:.3f}.png"
                        sign_path = os.path.join(save_dir, sign_filename)
                        sign_crop.save(sign_path)
                        
                        extracted_count += 1
                        
                        # Log für Analyse
                        extraction_log.append({
                            'original_file': filename,
                            'extracted_file': sign_filename,
                            'iou': iou,
                            'bbox_original': [x1_orig, y1_orig, x2_orig, y2_orig],
                            'bbox_normalized': bbox_pred[i].cpu().tolist()
                        })
                        
                        print(f"Extrahiert: {sign_filename} (IoU: {iou:.3f})")
    
    # Speichere Extraktions-Log
    log_path = os.path.join(save_dir, 'extraction_log.json')
    with open(log_path, 'w') as f:
        json.dump({
            'total_images': total_images,
            'extracted_signs': extracted_count,
            'success_rate': extracted_count / total_images if total_images > 0 else 0,
            'confidence_threshold': confidence_threshold,
            'extractions': extraction_log
        }, f, indent=2)
    
    print(f"\n=== EXTRAKTION ABGESCHLOSSEN ===")
    print(f"Gesamt verarbeitete Bilder: {total_images}")
    print(f"Erfolgreich extrahierte Schilder: {extracted_count}")
    print(f"Erfolgsrate: {extracted_count/total_images*100:.1f}%")
    print(f"Extraktion-Log: {log_path}")
    
    return extracted_count, extraction_log

def evaluate_localizer(model, test_loader, save_dir='localization_results'):
    """Evaluierung der reinen Lokalisierung"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    total_iou = 0.0
    total_samples = 0
    all_results = []
    
    # IoU Bins für Analyse
    iou_bins = [0.1, 0.3, 0.5, 0.7, 0.9]
    iou_counts = {threshold: 0 for threshold in iou_bins}
    
    with torch.no_grad():
        for i, (data, bbox_targets, filenames) in enumerate(test_loader):
            data = data.to(model.device)
            bbox_targets = bbox_targets.to(model.device)
            
            if torch.cuda.is_available():
                with autocast():
                    bbox_pred = model(data)
            else:
                bbox_pred = model(data)
            
            # IoU berechnen
            batch_ious = compute_iou(bbox_pred, bbox_targets)
            
            for j in range(data.size(0)):
                iou = batch_ious[j].item()
                total_iou += iou
                total_samples += 1
                
                # IoU Bins zählen
                for threshold in iou_bins:
                    if iou >= threshold:
                        iou_counts[threshold] += 1
                
                # Ergebnisse sammeln
                result = {
                    'filename': filenames[j],
                    'iou': iou,
                    'pred_bbox': bbox_pred[j].cpu().tolist(),
                    'target_bbox': bbox_targets[j].cpu().tolist()
                }
                all_results.append(result)
    
    # Metriken berechnen
    avg_iou = total_iou / total_samples if total_samples > 0 else 0
    
    print(f"\n=== LOKALISIERUNG EVALUATION ===")
    print(f"Durchschnittliche IoU: {avg_iou:.4f}")
    print(f"Getestete Samples: {total_samples}")
    
    for threshold in iou_bins:
        percentage = (iou_counts[threshold] / total_samples) * 100 if total_samples > 0 else 0
        print(f"IoU >= {threshold}: {iou_counts[threshold]}/{total_samples} ({percentage:.1f}%)")
    
    # Ergebnisse speichern
    results_summary = {
        'avg_iou': avg_iou,
        'total_samples': total_samples,
        'iou_thresholds': {str(k): v for k, v in iou_counts.items()},
        'detailed_results': all_results
    }
    
    with open(os.path.join(save_dir, 'localization_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    return avg_iou, all_results

def visualize_predictions(model, test_loader, save_dir='localization_results', num_visualizations=10):
    """Visualisiert Lokalisierungs-Vorhersagen"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib nicht verfügbar - überspringe Visualisierung")
        return
    
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    visualized = 0
    
    with torch.no_grad():
        for data, bbox_targets, filenames in test_loader:
            if visualized >= num_visualizations:
                break
            
            data = data.to(model.device)
            
            if torch.cuda.is_available():
                with autocast():
                    bbox_pred = model(data)
            else:
                bbox_pred = model(data)
            
            for i in range(data.size(0)):
                if visualized >= num_visualizations:
                    break
                
                filename = filenames[i]
                
                # Originalbild laden
                image_path = os.path.join(test_loader.dataset.images_dir, filename)
                image = Image.open(image_path).convert('RGB')
                
                # Koordinaten denormalisieren
                pred_bbox = bbox_pred[i].cpu() * torch.tensor([IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT])
                target_bbox = bbox_targets[i].cpu() * torch.tensor([IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT])
                
                # IoU berechnen
                iou = compute_iou(bbox_pred[i:i+1], bbox_targets[i:i+1]).item()
                
                # Visualization
                fig, ax = plt.subplots(1, figsize=(12, 8))
                ax.imshow(image.resize((IMAGE_WIDTH, IMAGE_HEIGHT)))
                
                # Ground Truth (grün)
                rect_gt = patches.Rectangle(
                    (target_bbox[0], target_bbox[1]), 
                    target_bbox[2] - target_bbox[0], 
                    target_bbox[3] - target_bbox[1],
                    linewidth=3, edgecolor='green', facecolor='none', label='Ground Truth'
                )
                ax.add_patch(rect_gt)
                
                # Prediction (rot)
                rect_pred = patches.Rectangle(
                    (pred_bbox[0], pred_bbox[1]), 
                    pred_bbox[2] - pred_bbox[0], 
                    pred_bbox[3] - pred_bbox[1],
                    linewidth=3, edgecolor='red', facecolor='none', label='Prediction'
                )
                ax.add_patch(rect_pred)
                
                ax.set_title(f'{filename} | IoU: {iou:.3f}', fontsize=14, weight='bold')
                ax.legend()
                ax.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'localization_{visualized:03d}_{filename}'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                visualized += 1
    
    print(f"Visualisierungen gespeichert: {save_dir}/localization_*.png")

def main():
    start_time = time.time()
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    if NUMPY_AVAILABLE:
        np.random.seed(42)
    
    print("=== REINES LOKALISIERUNGS-CNN MIT SCHILD-EXTRAKTION ===")
    
    monitor = PerformanceMonitor()
    
    # Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Verwende Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Transformationen
    if TORCHVISION_AVAILABLE:
        train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        print("Torchvision nicht verfügbar")
        train_transform = None
        test_transform = None
    
    # Lokalisierungs-Datasets laden
    print("\nLade GTSDB für reine Lokalisierung...")
    
    train_dataset = GTSDBLocalizationDataset(
        images_dir=r'C:\Users\timau\Desktop\Datensaetze\GTSDB\Train',
        labels_file=r'C:\Users\timau\Desktop\Datensaetze\GTSDB\Train\gt-train.txt',
        transform=train_transform,
        is_train=True
    )
    
    test_dataset = GTSDBLocalizationDataset(
        images_dir=r'C:\Users\timau\Desktop\Datensaetze\GTSDB\Test',
        labels_file=r'C:\Users\timau\Desktop\Datensaetze\GTSDB\gt-test.txt',
        transform=test_transform,
        is_train=False
    )
    
    # Train/Validation Split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_split, val_split = random_split(train_dataset, [train_size, val_size])
    
    # DataLoader
    train_loader = DataLoader(
        train_split, 
        batch_size=4,  # Für 1360x800 Auflösung
        shuffle=True, 
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_split, 
        batch_size=4, 
        shuffle=False, 
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Training Samples: {len(train_split)}")
    print(f"Validation Samples: {len(val_split)}")
    print(f"Test Samples: {len(test_dataset)}")
    print(f"Eingabeauflösung: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Extraktions-Verzeichnis: {EXTRACTED_SIGNS_DIR}")
    
    # Reines Lokalisierungs-Modell erstellen
    print("\n=== REINES LOKALISIERUNGS-MODELL ===")
    model = TrafficSignLocalizer().to(device)
    model.device = device
    
    total_params, trainable_params = model.get_trainable_parameters()
    print(f"Gesamt Parameter: {total_params:,}")
    print(f"Trainierbare Parameter: {trainable_params:,}")
    
    print("\n=== MODELL FEATURES ===")
    print("✓ Reine Bounding Box Regression (keine Klassifikation)")
    print("✓ Adaptive IoU + Smooth L1 Loss")
    print("✓ ECA-Net Channel Attention")
    print("✓ ReduceLROnPlateau Scheduler")
    print("✓ Mixed Precision Training")
    print("✓ Automatische Verkehrsschild-Extraktion")
    
    # Training
    learning_rate = 0.1  # Dramatisch höhere Learning Rate
    print(f"Learning Rate: {learning_rate} (aggressiv für Escape aus lokalem Minimum)")
    print("Starte reines Lokalisierungs-Training mit SGD + Momentum...")
    
    train_losses, val_losses, train_ious, val_ious = train_localizer(
        model, train_loader, val_loader, 
        num_epochs=50,
        lr=learning_rate, 
        monitor=monitor
    )
    
    # Bestes Modell laden
    print("\nLade bestes Modell für Evaluation...")
    model.load_state_dict(torch.load('best_traffic_sign_localizer.pth'))
    
    # Lokalisierung evaluieren
    print("\n=== LOKALISIERUNG EVALUATION ===")
    avg_iou, all_results = evaluate_localizer(model, test_loader)
    
    # VERKEHRSSCHILDER EXTRAKTION (Hauptfeature!)
    print("\n=== VERKEHRSSCHILDER EXTRAKTION ===")
    extracted_count, extraction_log = extract_traffic_signs(
        model, test_loader, 
        confidence_threshold=0.3  # Niedrigere Schwelle für mehr Extraktionen
    )
    
    # Visualisierung der Lokalisierung
    print("\n=== LOKALISIERUNGS-VISUALISIERUNG ===")
    visualize_predictions(model, test_loader, num_visualizations=10)
    
    # Performance-Test
    print("\n=== ECHTZEIT-PERFORMANCE TEST ===")
    model.eval()
    latencies = []
    
    # Warmup
    with torch.no_grad():
        for i, (data, _, _) in enumerate(test_loader):
            if i >= 3:
                break
            data = data.to(device)
            if torch.cuda.is_available():
                with autocast():
                    _ = model(data)
            else:
                _ = model(data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Latenz-Messung
    with torch.no_grad():
        for i, (data, _, _) in enumerate(test_loader):
            if i >= 20:
                break
            
            data = data.to(device)
            
            start_time = time.time()
            if torch.cuda.is_available():
                with autocast():
                    bbox_pred = model(data)
            else:
                bbox_pred = model(data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            batch_latency = (end_time - start_time) * 1000
            per_sample_latency = batch_latency / data.size(0)
            latencies.append(per_sample_latency)
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    throughput = 1000 / avg_latency if avg_latency > 0 else 0
    
    print(f"Durchschnittliche Latenz: {avg_latency:.2f} ms pro Bild")
    print(f"Durchsatz: {throughput:.1f} FPS")
    
    # Hardware Performance
    perf_summary = monitor.get_summary()
    print(f"\n=== HARDWARE-PERFORMANCE ===")
    print(f"CPU (Ø/Max): {perf_summary['avg_cpu']:.1f}% / {perf_summary['max_cpu']:.1f}%")
    print(f"RAM (Ø/Max): {perf_summary['avg_memory']:.1f}% / {perf_summary['max_memory']:.1f}%")
    print(f"GPU Memory (Ø/Max): {perf_summary['avg_gpu_memory']:.1f}% / {perf_summary['max_gpu_memory']:.1f}%")
    
    # Training Verlauf visualisieren
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(15, 5))
        
        # Loss Plot
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title('Lokalisierungs-Loss Verlauf')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # IoU Plot
        plt.subplot(1, 3, 2)
        plt.plot(train_ious, label='Training IoU', color='blue')
        plt.plot(val_ious, label='Validation IoU', color='red')
        plt.title('IoU Verlauf')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True)
        
        # IoU Distribution
        plt.subplot(1, 3, 3)
        test_ious = [result['iou'] for result in all_results]
        plt.hist(test_ious, bins=20, alpha=0.7, color='green')
        plt.axvline(avg_iou, color='red', linestyle='--', label=f'Durchschnitt: {avg_iou:.3f}')
        plt.title('IoU Verteilung (Test Set)')
        plt.xlabel('IoU')
        plt.ylabel('Anzahl')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pure_localization_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Training-Verlauf gespeichert: pure_localization_results.png")
    
    # Gesamtzeit
    total_time = time.time() - start_time
    print(f"\nGesamte Ausführungszeit: {total_time/60:.2f} Minuten")
    
    # Finales Summary
    print("\n" + "="*70)
    print("REINES LOKALISIERUNGS-CNN TRAINING ABGESCHLOSSEN")
    print("="*70)
    print(f"🎯 Finale Ergebnisse:")
    print(f"   • Durchschnittliche IoU: {avg_iou:.4f}")
    print(f"   • Extrahierte Verkehrsschilder: {extracted_count}")
    print(f"   • Inference-Geschwindigkeit: {throughput:.1f} FPS")
    print(f"   • Modell-Parameter: {total_params:,}")
    
    print(f"\n📁 Gespeicherte Dateien:")
    print(f"   • best_traffic_sign_localizer.pth")
    print(f"   • {EXTRACTED_SIGNS_DIR}/sign_*.png (extrahierte Schilder)")
    print(f"   • {EXTRACTED_SIGNS_DIR}/extraction_log.json")
    print(f"   • localization_results/localization_results.json")
    print(f"   • pure_localization_results.png")
    
    print(f"\n🚀 Kernfeatures:")
    print(f"   • Reine Bounding Box Lokalisierung")
    print(f"   • Automatische Verkehrsschild-Extraktion")
    print(f"   • ECA-Net Channel Attention")
    print(f"   • Adaptive Loss-Strategie")
    print(f"   • Ready für Pipeline mit GTSRB-Klassifikator")
    
    print(f"\n📊 Pipeline-Workflow:")
    print(f"   1. Vollbild → Lokalisierungs-CNN → Bounding Box")
    print(f"   2. Bounding Box → Crop → Verkehrsschild-Bild")
    print(f"   3. Verkehrsschild-Bild → GTSRB-Klassifikator → Klasse")
    print(f"   4. Result: Lokalisierung + Klassifikation")
    
    # Erfolgs-Assessment
    extraction_rate = extracted_count / len(test_dataset) if len(test_dataset) > 0 else 0
    
    if avg_iou > 0.5 and extraction_rate > 0.6:
        print(f"\n✅ Ausgezeichnete Lokalisierung! Pipeline-ready.")
        print(f"   • IoU > 0.5 erreicht")
        print(f"   • Hohe Extraktionsrate: {extraction_rate*100:.1f}%")
    elif avg_iou > 0.3 and extraction_rate > 0.4:
        print(f"\n✅ Gute Lokalisierung für Maturaarbeit!")
        print(f"   • Solide IoU: {avg_iou:.3f}")
        print(f"   • Extraktionsrate: {extraction_rate*100:.1f}%")
    else:
        print(f"\n⚠️ Verbesserungspotential:")
        print(f"   • IoU könnte höher sein (aktuell: {avg_iou:.3f})")
        print(f"   • Extraktionsrate: {extraction_rate*100:.1f}%")
        print(f"   • Eventuell längeres Training oder Hyperparameter-Tuning")
    
    print(f"\n💡 Nächste Schritte für Pipeline:")
    print(f"   • Lade Ihr GTSRB-Klassifikationsmodell")
    print(f"   • Teste extrahierte Schilder: {EXTRACTED_SIGNS_DIR}")
    print(f"   • Erstelle End-to-End Pipeline-Skript")
    print(f"   • Vergleiche Performance: Pipeline vs End-to-End")

if __name__ == "__main__":
    main()