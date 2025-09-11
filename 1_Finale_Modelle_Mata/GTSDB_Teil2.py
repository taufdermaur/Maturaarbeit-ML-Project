import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import psutil
from PIL import Image
import os
import json
import csv
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importiere optional
try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Torchvision nicht verfügbar")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib nicht verfügbar")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except:
    GPUTIL_AVAILABLE = False

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

# CNN Modell für Klassifikation (identisch mit GTSRB-Training)
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

class DetectionDataset(Dataset):
    """Dataset für Bilder mit Bounding Box Koordinaten"""
    
    def __init__(self, detections_file, images_dir, transform=None):
        """
        detections_file: JSON oder CSV mit Format:
        [{"filename": "00600.png", "boxes": [[x1, y1, x2, y2, conf], ...]}, ...]
        oder CSV: filename,x1,y1,x2,y2,confidence
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.detections = []
        
        # Lade Detektionen
        if detections_file.endswith('.json'):
            with open(detections_file, 'r') as f:
                self.detections = json.load(f)
        elif detections_file.endswith('.csv'):
            self.detections = self._load_csv(detections_file)
        else:
            raise ValueError("Detections file muss .json oder .csv sein")
        
        print(f"Geladene Detektionen: {len(self.detections)} Bilder")
    
    def _load_csv(self, csv_file):
        """Lädt CSV Format: filename,x1,y1,x2,y2,confidence"""
        detections = {}
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 6:
                    filename = row[0]
                    x1, y1, x2, y2, conf = map(float, row[1:6])
                    
                    if filename not in detections:
                        detections[filename] = {'filename': filename, 'boxes': []}
                    detections[filename]['boxes'].append([x1, y1, x2, y2, conf])
        
        return list(detections.values())
    
    def __len__(self):
        return len(self.detections)
    
    def __getitem__(self, idx):
        detection = self.detections[idx]
        filename = detection['filename']
        boxes = detection['boxes']
        
        # Bild laden
        image_path = self.images_dir / filename
        if not image_path.exists():
            # Versuche verschiedene Endungen
            for ext in ['.png', '.jpg', '.jpeg', '.ppm']:
                alt_path = image_path.with_suffix(ext)
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        
        # Extrahiere alle Patches
        patches = []
        valid_boxes = []
        
        for box in boxes:
            x1, y1, x2, y2, conf = box
            
            # Überprüfe Koordinatenformat (normalisiert vs. Pixel)
            if max(x1, y1, x2, y2) <= 1.0:
                # Normalisierte Koordinaten -> zu Pixeln
                x1_pixel = int(x1 * width)
                y1_pixel = int(y1 * height)
                x2_pixel = int(x2 * width)
                y2_pixel = int(y2 * height)
            else:
                # Bereits Pixelkoordinaten
                x1_pixel = int(x1)
                y1_pixel = int(y1)
                x2_pixel = int(x2)
                y2_pixel = int(y2)
            
            # Validiere Bounding Box
            if x2_pixel > x1_pixel and y2_pixel > y1_pixel:
                # Patch extrahieren
                patch = image.crop((x1_pixel, y1_pixel, x2_pixel, y2_pixel))
                
                # Auf 64x64 resizen
                patch = patch.resize((64, 64), Image.LANCZOS)
                
                if self.transform:
                    patch = self.transform(patch)
                
                patches.append(patch)
                valid_boxes.append([x1_pixel, y1_pixel, x2_pixel, y2_pixel, conf])
        
        return {
            'filename': filename,
            'image': image,
            'patches': patches,
            'boxes': valid_boxes
        }

def load_classification_model(model_path, num_classes=43, device='cuda'):
    """Lädt das trainierte Klassifikationsmodell"""
    model = ClassificationCNN(num_classes).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Klassifikationsmodell geladen von: {model_path}")
        return model
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        return None

def classify_patches(model, patches, device='cuda'):
    """Klassifiziert eine Liste von Patches"""
    if not patches:
        return []
    
    # Patches zu Tensor
    batch = torch.stack(patches).to(device)
    
    with torch.no_grad():
        outputs = model(batch)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
    
    results = []
    for i in range(len(patches)):
        results.append({
            'class_id': predictions[i].item(),
            'confidence': confidences[i].item(),
            'probabilities': probabilities[i].cpu().numpy()
        })
    
    return results

def visualize_results(image, boxes, classifications, save_path=None):
    """Visualisiert Ergebnisse mit Matplotlib"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib nicht verfügbar - Visualisierung übersprungen")
        return
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    
    for i, (box, classification) in enumerate(zip(boxes, classifications)):
        x1, y1, x2, y2, detection_conf = box
        class_id = classification['class_id']
        class_conf = classification['confidence']
        
        # Rechteck zeichnen
        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Label mit Klasse und Confidence
        label = f'Klasse {class_id}\nConf: {class_conf:.3f}'
        ax.text(x1, y1-10, label, color=color, fontsize=10, 
               weight='bold', bbox=dict(boxstyle='round,pad=0.3', 
               facecolor='white', alpha=0.8))
    
    ax.set_title(f'Klassifikationsergebnisse ({len(classifications)} Schilder)', 
                fontsize=14, weight='bold')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_results_csv(results, output_file):
    """Speichert Ergebnisse in CSV"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'box_x1', 'box_y1', 'box_x2', 'box_y2', 
                        'detection_confidence', 'predicted_class', 'classification_confidence'])
        
        for result in results:
            filename = result['filename']
            for box, classification in zip(result['boxes'], result['classifications']):
                x1, y1, x2, y2, det_conf = box
                class_id = classification['class_id']
                class_conf = classification['confidence']
                
                writer.writerow([filename, x1, y1, x2, y2, det_conf, class_id, class_conf])

def process_batch(model, dataset, device, output_dir, max_visualizations=10):
    """Verarbeitet einen Batch von Detektionen"""
    results = []
    visualizations_created = 0
    
    for i in range(len(dataset)):
        data = dataset[i]
        filename = data['filename']
        image = data['image']
        patches = data['patches']
        boxes = data['boxes']
        
        if not patches:
            print(f"Keine gültigen Patches in {filename}")
            continue
        
        # Klassifiziere Patches
        classifications = classify_patches(model, patches, device)
        
        # Ergebnis speichern
        results.append({
            'filename': filename,
            'boxes': boxes,
            'classifications': classifications
        })
        
        # Visualisierung (nur für erste Bilder)
        if MATPLOTLIB_AVAILABLE and visualizations_created < max_visualizations:
            vis_path = Path(output_dir) / f'classification_{filename}'
            visualize_results(image, boxes, classifications, vis_path)
            visualizations_created += 1
        
        # Progress
        if (i + 1) % 10 == 0:
            print(f"Verarbeitet: {i + 1}/{len(dataset)} Bilder")
    
    return results

def realtime_evaluation(model, dataset, device, warmup_samples=10):
    """Misst Echtzeit-Performance der Klassifikation"""
    model.eval()
    
    # Warmup
    print("Warmup...")
    for i in range(min(warmup_samples, len(dataset))):
        data = dataset[i]
        patches = data['patches']
        if patches:
            classify_patches(model, patches, device)
    
    # Performance-Messung
    print("Messe Performance...")
    latencies = []
    total_patches = 0
    max_samples = min(100, len(dataset))
    
    for i in range(max_samples):
        data = dataset[i]
        patches = data['patches']
        
        if patches:
            start_time = time.time()
            classify_patches(model, patches, device)
            end_time = time.time()
            
            latency_per_patch = ((end_time - start_time) / len(patches)) * 1000  # ms
            latencies.extend([latency_per_patch] * len(patches))
            total_patches += len(patches)
    
    if latencies:
        mean_latency = sum(latencies) / len(latencies)
        throughput = 1000 / mean_latency if mean_latency > 0 else 0
        
        print(f"\n=== KLASSIFIKATION PERFORMANCE ===")
        print(f"Durchschnittliche Latenz pro Patch: {mean_latency:.2f} ms")
        print(f"Durchsatz: {throughput:.1f} Patches/s")
        print(f"Verarbeitete Patches: {total_patches}")
    
    return latencies

def main():
    # Timer starten
    start_time = time.time()
    
    # Seed für Reproduzierbarkeit
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Performance Monitor
    monitor = PerformanceMonitor()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Verwende Device: {device}")
    
    # Transformationen für 64x64 Patches
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Pfade definieren (anpassen!)
    DETECTIONS_FILE = 'detections.json'  # oder .csv
    IMAGES_DIR = r'C:\Users\timau\Desktop\Datensaetze\GTSDB\Test'
    MODEL_PATH = r'C:\Users\timau\Desktop\gtsrb_model.pth'
    OUTPUT_DIR = 'classification_results'
    
    # Output-Ordner erstellen
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    print("=== VERKEHRSSCHILDKLASSIFIKATION PIPELINE ===")
    
    # Klassifikationsmodell laden
    print("\n1. Lade Klassifikationsmodell...")
    model = load_classification_model(MODEL_PATH, num_classes=43, device=device)
    if model is None:
        print("Kann nicht ohne Modell fortfahren!")
        return
    
    # Dataset laden
    print("\n2. Lade Detektionsdaten...")
    try:
        dataset = DetectionDataset(DETECTIONS_FILE, IMAGES_DIR, transform=transform)
    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
        print("Stelle sicher, dass die Detektionsdatei existiert!")
        return
    
    # Performance logging
    monitor.log_performance()
    
    # Klassifikation durchführen
    print("\n3. Führe Klassifikation durch...")
    results = process_batch(model, dataset, device, OUTPUT_DIR, max_visualizations=10)
    
    # Ergebnisse speichern
    print("\n4. Speichere Ergebnisse...")
    results_csv = Path(OUTPUT_DIR) / 'classification_results.csv'
    save_results_csv(results, results_csv)
    
    # Performance evaluation
    print("\n5. Performance-Evaluation...")
    latencies = realtime_evaluation(model, dataset, device)
    
    # Statistiken
    total_images = len(results)
    total_patches = sum(len(r['classifications']) for r in results)
    
    print(f"\n=== ZUSAMMENFASSUNG ===")
    print(f"Verarbeitete Bilder: {total_images}")
    print(f"Klassifizierte Patches: {total_patches}")
    print(f"Durchschnittliche Patches pro Bild: {total_patches/total_images if total_images > 0 else 0:.1f}")
    
    # Hardware-Performance
    perf_summary = monitor.get_summary()
    print(f"\n=== HARDWARE-PERFORMANCE ===")
    print(f"CPU: {perf_summary['avg_cpu']:.1f}% (Max: {perf_summary['max_cpu']:.1f}%)")
    print(f"RAM: {perf_summary['avg_memory']:.1f}% (Max: {perf_summary['max_memory']:.1f}%)")
    print(f"GPU Memory: {perf_summary['avg_gpu_memory']:.1f}% (Max: {perf_summary['max_gpu_memory']:.1f}%)")
    
    # Gesamtzeit
    total_time = time.time() - start_time
    print(f"\nGesamte Ausführungszeit: {total_time:.2f} Sekunden")
    
    print(f"\n=== AUSGABEDATEIEN ===")
    print(f"- Ergebnisse: {results_csv}")
    print(f"- Visualisierungen: {OUTPUT_DIR}/classification_*.png")
    
    # Beispiel für Detektionsdatei-Format
    print(f"\n=== HINWEIS ZUR DETEKTIONSDATEI ===")
    print("Die Detektionsdatei sollte folgendes Format haben:")
    print("JSON: [{'filename': '00600.png', 'boxes': [[x1,y1,x2,y2,conf], ...]}, ...]")
    print("CSV: filename,x1,y1,x2,y2,confidence (eine Zeile pro Box)")

if __name__ == "__main__":
    main()