import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import time
import psutil
from PIL import Image, ImageDraw
import os
import warnings
warnings.filterwarnings('ignore')

# Importiere nur benötigte Teile von torchvision
try:
    from torchvision import transforms
    print("Torchvision erfolgreich importiert")
except ImportError as e:
    print(f"Torchvision Import Fehler: {e}")
    # Fallback für transforms
    import torch.nn.functional as F

# Importiere optional
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except:
    NUMPY_AVAILABLE = False
    print("NumPy nicht verfügbar - verwende nur PyTorch")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib nicht verfügbar")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except:
    GPUTIL_AVAILABLE = False
    print("GPUtil nicht verfügbar")

try:
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False
    print("Sklearn nicht verfügbar")

# Globale Konstanten
MAX_DETECTIONS = 5  # Maximale Anzahl Bounding Boxes pro Bild
IMAGE_WIDTH = 1360
IMAGE_HEIGHT = 800
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3

# Performance Monitoring (vereinfacht)
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
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            gpu_memory = gpu.memoryUtil * 100
        
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_percent)
        self.gpu_memory_history.append(gpu_memory)
        self.timestamps.append(time.time())
    
    def get_summary(self):
        return {
            'avg_cpu': np.mean(self.cpu_history) if self.cpu_history else 0,
            'avg_memory': np.mean(self.memory_history) if self.memory_history else 0,
            'avg_gpu_memory': np.mean(self.gpu_memory_history) if self.gpu_memory_history else 0,
            'max_cpu': np.max(self.cpu_history) if self.cpu_history else 0,
            'max_memory': np.max(self.memory_history) if self.memory_history else 0,
            'max_gpu_memory': np.max(self.gpu_memory_history) if self.gpu_memory_history else 0,
        }

# Timer
start_time = time.time()

# Seed für Reproduzierbarkeit
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Performance Monitor initialisieren
monitor = PerformanceMonitor()

# Device definieren
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Verwende Device: {device}")

# Transformationen für Lokalisation
localization_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformationen für Klassifikation (64x64)
classification_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class GTSDBDataset(Dataset):
    """Dataset für GTSDB Bilder mit Bounding Box Labels"""
    
    def __init__(self, images_dir, labels_file, transform=None, is_train=True):
        self.images_dir = images_dir
        self.transform = transform
        self.is_train = is_train
        
        # Labels laden
        self.annotations = []
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) == 6:
                    filename = parts[0].replace('.ppm', '.png')  # PNG statt PPM
                    x1, y1, x2, y2, class_id = map(int, parts[1:])
                    self.annotations.append({
                        'filename': filename,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'class_id': class_id
                    })
        
        # Gruppiere Annotationen nach Dateiname
        self.grouped_annotations = {}
        for ann in self.annotations:
            filename = ann['filename']
            if filename not in self.grouped_annotations:
                self.grouped_annotations[filename] = []
            self.grouped_annotations[filename].append(ann)
        
        self.image_files = list(self.grouped_annotations.keys())
        print(f"Geladene Bilder: {len(self.image_files)}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = os.path.join(self.images_dir, filename)
        
        # Bild laden
        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size
        
        # Bild transformieren
        if self.transform:
            image = self.transform(image)
        
        # Labels für dieses Bild
        annotations = self.grouped_annotations[filename]
        
        # Target erstellen: [MAX_DETECTIONS, 5] -> [x1, y1, x2, y2, confidence]
        target = torch.zeros(MAX_DETECTIONS, 5)
        
        for i, ann in enumerate(annotations[:MAX_DETECTIONS]):
            # Koordinaten normalisieren auf [0, 1]
            x1_norm = ann['x1'] / original_width
            y1_norm = ann['y1'] / original_height
            x2_norm = ann['x2'] / original_width
            y2_norm = ann['y2'] / original_height
            
            target[i] = torch.tensor([x1_norm, y1_norm, x2_norm, y2_norm, 1.0])
        
        return image, target, filename

class TrafficSignLocalizationCNN(nn.Module):
    """Custom CNN für Verkehrsschildlokalisation"""
    
    def __init__(self, max_detections=MAX_DETECTIONS):
        super(TrafficSignLocalizationCNN, self).__init__()
        self.max_detections = max_detections
        
        # Feature Extractor mit kontrolliertem Downsampling
        self.backbone = nn.Sequential(
            # Block 1: 1360x800 -> 680x400
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 680x400 -> 340x200
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 340x200 -> 170x100
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 170x100 -> 85x50
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5: 85x50 -> 43x25 (etwa)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Global Average Pooling + Klassifikator
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Detection Head: 5 Boxen x 5 Werte (x1, y1, x2, y2, confidence)
        self.detection_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, max_detections * 5)  # 5 boxes * 5 values
        )
        
        # Koordinaten auf [0,1] begrenzen, Confidence durch Sigmoid
        self.coordinate_activation = nn.Sigmoid()
    
    def forward(self, x):
        # Feature Extraction
        features = self.backbone(x)
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Detection
        detection_output = self.detection_head(pooled)
        
        # Reshape zu [batch_size, max_detections, 5]
        detection_output = detection_output.view(-1, self.max_detections, 5)
        
        # Aktivierungen anwenden
        coordinates = self.coordinate_activation(detection_output[:, :, :4])
        confidence = torch.sigmoid(detection_output[:, :, 4:5])
        
        output = torch.cat([coordinates, confidence], dim=2)
        
        return output

class ClassificationCNN(nn.Module):
    """CNN für Verkehrsschildklassifikation (64x64 -> Klasse)"""
    
    def __init__(self, num_classes):
        super(ClassificationCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
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

def compute_iou(box1, box2):
    """Berechnet IoU zwischen zwei Bounding Boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def localization_loss(predictions, targets):
    """
    Loss für Multi-Object Detection
    predictions: [batch_size, max_detections, 5]
    targets: [batch_size, max_detections, 5]
    """
    batch_size = predictions.size(0)
    total_loss = 0.0
    
    for b in range(batch_size):
        pred = predictions[b]  # [max_detections, 5]
        target = targets[b]    # [max_detections, 5]
        
        # Confidence Loss (BCE)
        conf_loss = nn.functional.binary_cross_entropy(
            pred[:, 4], target[:, 4], reduction='sum'
        )
        
        # Coordinate Loss (nur für positive Targets)
        positive_mask = target[:, 4] > 0.5
        if positive_mask.sum() > 0:
            coord_loss = nn.functional.mse_loss(
                pred[positive_mask, :4], target[positive_mask, :4], reduction='sum'
            )
        else:
            coord_loss = 0.0
        
        total_loss += conf_loss + coord_loss
    
    return total_loss / batch_size

def train_localization_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    """Training der Lokalisations-CNN"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        if epoch % 5 == 0:
            monitor.log_performance()
        
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (data, targets, _) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(data)
            loss = localization_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, targets, _ in val_loader:
                data, targets = data.to(device), targets.to(device)
                predictions = model(data)
                loss = localization_loss(predictions, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_localization_model.pth')
        
        scheduler.step()
    
    return train_losses, val_losses

def extract_sign_patches(image, detections, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Extrahiert 64x64 Patches von detektierten Schildern
    image: PIL Image (original)
    detections: [max_detections, 5] numpy array
    """
    patches = []
    valid_boxes = []
    
    width, height = image.size
    
    for detection in detections:
        x1, y1, x2, y2, conf = detection
        
        if conf > confidence_threshold:
            # Koordinaten zurück zu Pixeln
            x1_pixel = int(x1 * width)
            y1_pixel = int(y1 * height)
            x2_pixel = int(x2 * width)
            y2_pixel = int(y2 * height)
            
            # Patch extrahieren
            patch = image.crop((x1_pixel, y1_pixel, x2_pixel, y2_pixel))
            patch = patch.resize((64, 64))
            patches.append(patch)
            valid_boxes.append([x1_pixel, y1_pixel, x2_pixel, y2_pixel])
    
    return patches, valid_boxes

def visualize_detections(image, detections, save_path, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Visualisiert Detektionen mit grünen Rechtecken"""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    width, height = image.size
    detection_count = 0
    
    for detection in detections:
        x1, y1, x2, y2, conf = detection
        
        if conf > confidence_threshold:
            # Koordinaten zurück zu Pixeln
            x1_pixel = x1 * width
            y1_pixel = y1 * height
            x2_pixel = x2 * width
            y2_pixel = y2 * height
            
            # Rechteck zeichnen
            rect = patches.Rectangle(
                (x1_pixel, y1_pixel), 
                x2_pixel - x1_pixel, 
                y2_pixel - y1_pixel,
                linewidth=2, 
                edgecolor='green', 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Confidence anzeigen
            ax.text(x1_pixel, y1_pixel-5, f'{conf:.2f}', 
                   color='green', fontsize=12, weight='bold')
            detection_count += 1
    
    ax.set_title(f'Detected Traffic Signs ({detection_count} found)', fontsize=14, weight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_localization(model, test_loader, save_dir='evaluation_results'):
    """Evaluiert das Lokalisationsmodell"""
    model.eval()
    total_detections = 0
    total_ground_truth = 0
    correct_detections = 0
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (data, targets, filenames) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            predictions = model(data)
            
            # Für jedes Bild in Batch
            for j in range(data.size(0)):
                pred = predictions[j].cpu().numpy()  # [max_detections, 5]
                target = targets[j].cpu().numpy()   # [max_detections, 5]
                filename = filenames[j]
                
                # Anzahl Ground Truth Objekte
                gt_count = (target[:, 4] > 0.5).sum()
                total_ground_truth += gt_count
                
                # Anzahl Detektionen
                det_count = (pred[:, 4] > CONFIDENCE_THRESHOLD).sum()
                total_detections += det_count
                
                # IoU Matching für korrekte Detektionen
                for pred_box in pred:
                    if pred_box[4] > CONFIDENCE_THRESHOLD:
                        best_iou = 0
                        for gt_box in target:
                            if gt_box[4] > 0.5:
                                iou = compute_iou(pred_box[:4], gt_box[:4])
                                best_iou = max(best_iou, iou)
                        
                        if best_iou > IOU_THRESHOLD:
                            correct_detections += 1
                
                # Visualisierung für erste 10 Bilder
                if i * data.size(0) + j < 10:
                    # Original Bild laden
                    image_path = os.path.join(test_loader.dataset.images_dir, filename)
                    image = Image.open(image_path).convert('RGB')
                    
                    save_path = os.path.join(save_dir, f'detection_{filename}')
                    visualize_detections(image, pred, save_path)
    
    # Metriken berechnen
    precision = correct_detections / total_detections if total_detections > 0 else 0
    recall = correct_detections / total_ground_truth if total_ground_truth > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n=== LOKALISATION EVALUATION ===")
    print(f"Total Ground Truth: {total_ground_truth}")
    print(f"Total Detections: {total_detections}")
    print(f"Correct Detections: {correct_detections}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return precision, recall, f1

def realtime_evaluation(localization_model, classification_model, test_loader, warmup_batches=5):
    """Echtzeit-Evaluation für gesamte Pipeline"""
    localization_model.eval()
    if classification_model:
        classification_model.eval()
    
    # Warmup
    print("Warmup...")
    with torch.no_grad():
        for i, (data, _, _) in enumerate(test_loader):
            if i >= warmup_batches:
                break
            data = data.to(device)
            _ = localization_model(data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Latenz-Messung
    latencies = []
    total_samples = 0
    max_samples = 100
    
    with torch.no_grad():
        for data, _, filenames in test_loader:
            for j in range(data.size(0)):
                if total_samples >= max_samples:
                    break
                
                single_image = data[j:j+1].to(device)
                
                start_time = time.time()
                
                # Lokalisation
                detections = localization_model(single_image)
                
                # Falls Klassifikation aktiviert
                if classification_model:
                    # TODO: Hier würden die extrahierten Patches klassifiziert
                    pass
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                total_samples += 1
            
            if total_samples >= max_samples:
                break
    
    mean_latency = sum(latencies) / len(latencies) if latencies else 0
    throughput = 1000 / mean_latency if mean_latency > 0 else 0  # FPS
    
    print(f"\n=== ECHTZEIT-PERFORMANCE ===")
    print(f"Durchschnittliche Latenz: {mean_latency:.2f} ms")
    print(f"Durchsatz: {throughput:.1f} FPS")
    
    return latencies, throughput

# === HAUPTPROGRAMM ===

if __name__ == "__main__":
    # Datensets laden
    print("Lade GTSDB Datensätze...")
    
    train_dataset = GTSDBDataset(
        images_dir=r'C:\Users\timau\Desktop\Datensaetze\GTSDB\Train',
        labels_file=r'C:\Users\timau\Desktop\Datensaetze\GTSDB\Train\gt-train.txt',
        transform=localization_transform,
        is_train=True
    )
    
    test_dataset = GTSDBDataset(
        images_dir=r'C:\Users\timau\Desktop\Datensaetze\GTSDB\Test',
        labels_file=r'C:\Users\timau\Desktop\Datensaetze\GTSDB\gt-test.txt',
        transform=localization_transform,
        is_train=False
    )
    
    # Train/Validation Split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_split, val_split = random_split(train_dataset, [train_size, val_size])
    
    # DataLoader
    train_loader = DataLoader(train_split, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_split, batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"Training Samples: {len(train_split)}")
    print(f"Validation Samples: {len(val_split)}")
    print(f"Test Samples: {len(test_dataset)}")
    
    # Lokalisationsmodell erstellen und trainieren
    print("\n=== LOKALISATIONSMODELL TRAINING ===")
    localization_model = TrafficSignLocalizationCNN(max_detections=MAX_DETECTIONS).to(device)
    
    total_params = sum(p.numel() for p in localization_model.parameters())
    print(f"Modell Parameter: {total_params:,}")
    
    # Training starten
    train_losses, val_losses = train_localization_model(
        localization_model, train_loader, val_loader, num_epochs=30, lr=0.001
    )
    
    # Bestes Modell laden
    localization_model.load_state_dict(torch.load('best_localization_model.pth'))
    
    # Evaluation
    print("\n=== LOKALISATION EVALUATION ===")
    precision, recall, f1 = evaluate_localization(localization_model, test_loader)
    
    # Klassifikationsmodell laden (bereits trainiert)
    print("\n=== KLASSIFIKATIONSMODELL LADEN ===")
    try:
        classification_model = ClassificationCNN(num_classes=43).to(device)  # GTSRB hat 43 Klassen
        classification_model.load_state_dict(torch.load(r'C:\Users\timau\Desktop\gtsrb_model.pth'))
        print("Klassifikationsmodell erfolgreich geladen")
    except:
        print("Klassifikationsmodell nicht gefunden - nur Lokalisation wird evaluiert")
        classification_model = None
    
    # Echtzeit-Evaluation
    print("\n=== ECHTZEIT-EVALUATION ===")
    latencies, throughput = realtime_evaluation(localization_model, classification_model, test_loader)
    
    # Performance Summary
    perf_summary = monitor.get_summary()
    print(f"\n=== HARDWARE-PERFORMANCE ZUSAMMENFASSUNG ===")
    print(f"CPU (Durchschnitt/Max): {perf_summary['avg_cpu']:.1f}% / {perf_summary['max_cpu']:.1f}%")
    print(f"RAM (Durchschnitt/Max): {perf_summary['avg_memory']:.1f}% / {perf_summary['max_memory']:.1f}%")
    print(f"GPU Memory (Durchschnitt/Max): {perf_summary['avg_gpu_memory']:.1f}% / {perf_summary['max_gpu_memory']:.1f}%")
    
    # Gesamtzeit
    total_time = time.time() - start_time
    print(f"\nGesamte Ausführungszeit: {total_time/60:.2f} Minuten")
    
    print("\n=== TRAINING ABGESCHLOSSEN ===")
    print("Gespeicherte Dateien:")
    print("- best_localization_model.pth")
    print("- evaluation_results/detection_*.png")