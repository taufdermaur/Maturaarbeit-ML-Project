import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import cv2
import os
import random
from PIL import Image
import time
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import json
from datetime import datetime

# ==========================
# ÜBERWACHUNGS-IMPORTS
# ==========================
import psutil
import threading
from datetime import datetime
try:
    import GPUtil
    GPU_VERFUEGBAR = True
except ImportError:
    GPU_VERFUEGBAR = False
    print("GPUtil nicht verfügbar - GPU-Überwachung deaktiviert")

# ==========================
# DATENSAMMLER-KLASSE
# ==========================
class DatenSammler:
    def __init__(self):
        self.alle_daten = {
            'allgemeine_info': {},
            'training_metriken': {
                'epochen': [],
                'train_loss': [],
                'train_accuracy': [],
                'val_loss': [],
                'val_accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': []
            },
            'pipeline_testmetriken': [],
            'einzelbild_ergebnisse': [],
            'leistungsmetriken': {
                'inferenzzeiten': [],
                'erkennungskonfidenzen': [],
                'klassifikationskonfidenzen': [],
                'batch_benchmarks': {}
            },
            'ressourcennutzung': {
                'zeitstempel': [],
                'cpu_prozent': [],
                'ram_prozent': [],
                'ram_verwendet_gb': [],
                'gpu_prozent': [],
                'gpu_speicher_prozent': [],
                'gpu_speicher_verwendet_gb': []
            },
            'zeitmessungen': {}
        }
    
    def zu_excel_exportieren(self, dateiname):
        """Alle gesammelten Daten in Excel-Datei exportieren"""
        with pd.ExcelWriter(dateiname, engine='openpyxl') as writer:
            
            # Allgemeine Informationen
            allgemeine_df = pd.DataFrame([self.alle_daten['allgemeine_info']])
            allgemeine_df.to_excel(writer, sheet_name='Allgemeine_Info', index=False)
            
            # Training Metriken
            if self.alle_daten['training_metriken']['epochen']:
                training_df = pd.DataFrame(self.alle_daten['training_metriken'])
                training_df.to_excel(writer, sheet_name='Training_Metriken', index=False)
            
            # Pipeline Test Ergebnisse
            if self.alle_daten['pipeline_testmetriken']:
                pipeline_test_df = pd.DataFrame(self.alle_daten['pipeline_testmetriken'])
                pipeline_test_df.to_excel(writer, sheet_name='Pipeline_Tests', index=False)
            
            # Einzelbild-Ergebnisse
            if self.alle_daten['einzelbild_ergebnisse']:
                einzelbild_df = pd.DataFrame(self.alle_daten['einzelbild_ergebnisse'])
                einzelbild_df.to_excel(writer, sheet_name='Einzelbild_Ergebnisse', index=False)
            
            # Leistungsmetriken - Inferenzzeiten
            if self.alle_daten['leistungsmetriken']['inferenzzeiten']:
                inferenz_df = pd.DataFrame({
                    'inferenzzeit_ms': self.alle_daten['leistungsmetriken']['inferenzzeiten'],
                    'erkennungskonfidenz': self.alle_daten['leistungsmetriken']['erkennungskonfidenzen'] 
                        if self.alle_daten['leistungsmetriken']['erkennungskonfidenzen'] else [None] * len(self.alle_daten['leistungsmetriken']['inferenzzeiten']),
                    'klassifikationskonfidenz': self.alle_daten['leistungsmetriken']['klassifikationskonfidenzen']
                        if self.alle_daten['leistungsmetriken']['klassifikationskonfidenzen'] else [None] * len(self.alle_daten['leistungsmetriken']['inferenzzeiten'])
                })
                inferenz_df.to_excel(writer, sheet_name='Inferenz_Zeiten', index=False)
            
            # Batch-Benchmarks
            if self.alle_daten['leistungsmetriken']['batch_benchmarks']:
                benchmark_daten = []
                for batch_groesse, metriken in self.alle_daten['leistungsmetriken']['batch_benchmarks'].items():
                    zeile = {'batch_groesse': batch_groesse}
                    zeile.update(metriken)
                    benchmark_daten.append(zeile)
                benchmark_df = pd.DataFrame(benchmark_daten)
                benchmark_df.to_excel(writer, sheet_name='Batch_Benchmarks', index=False)
            
            # Ressourcennutzung
            if self.alle_daten['ressourcennutzung']['zeitstempel']:
                ressourcen_df = pd.DataFrame(self.alle_daten['ressourcennutzung'])
                ressourcen_df.to_excel(writer, sheet_name='Ressourcennutzung', index=False)
            
            # Zeitmessungen
            if self.alle_daten['zeitmessungen']:
                zeit_df = pd.DataFrame([self.alle_daten['zeitmessungen']])
                zeit_df.to_excel(writer, sheet_name='Zeitmessungen', index=False)
        
        print(f"Alle Daten erfolgreich in Excel-Datei exportiert: {dateiname}")

# ==========================
# ÜBERWACHUNGSKLASSEN
# ==========================
class RessourcenUeberwachung:
    def __init__(self, daten_sammler):
        self.ueberwachung_aktiv = False
        self.daten_sammler = daten_sammler
        self.startzeit = None
        self.endzeit = None
        
    def ueberwachung_starten(self):
        """Ressourcenüberwachung starten"""
        self.ueberwachung_aktiv = True
        self.startzeit = time.time()
        print(f"Ressourcenüberwachung gestartet um {datetime.now().strftime('%H:%M:%S')}")
        
        self.ueberwachung_thread = threading.Thread(target=self._ressourcen_ueberwachen)
        self.ueberwachung_thread.daemon = True
        self.ueberwachung_thread.start()
        
    def ueberwachung_stoppen(self):
        """Ressourcenüberwachung stoppen"""
        self.ueberwachung_aktiv = False
        self.endzeit = time.time()
        print(f"Ressourcenüberwachung gestoppt um {datetime.now().strftime('%H:%M:%S')}")
        
    def _ressourcen_ueberwachen(self):
        """Überwachungsschleife"""
        while self.ueberwachung_aktiv:
            try:
                aktuelle_zeit = time.time() - self.startzeit
                self.daten_sammler.alle_daten['ressourcennutzung']['zeitstempel'].append(aktuelle_zeit)
                
                # CPU-Überwachung
                cpu_prozent = psutil.cpu_percent(interval=0.1)
                self.daten_sammler.alle_daten['ressourcennutzung']['cpu_prozent'].append(cpu_prozent)
                
                # RAM-Überwachung
                ram = psutil.virtual_memory()
                self.daten_sammler.alle_daten['ressourcennutzung']['ram_prozent'].append(ram.percent)
                self.daten_sammler.alle_daten['ressourcennutzung']['ram_verwendet_gb'].append(ram.used / (1024**3))
                
                # GPU-Überwachung
                if GPU_VERFUEGBAR:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            self.daten_sammler.alle_daten['ressourcennutzung']['gpu_prozent'].append(gpu.load * 100)
                            self.daten_sammler.alle_daten['ressourcennutzung']['gpu_speicher_prozent'].append(gpu.memoryUtil * 100)
                            self.daten_sammler.alle_daten['ressourcennutzung']['gpu_speicher_verwendet_gb'].append(gpu.memoryUsed / 1024)
                        else:
                            self.daten_sammler.alle_daten['ressourcennutzung']['gpu_prozent'].append(0)
                            self.daten_sammler.alle_daten['ressourcennutzung']['gpu_speicher_prozent'].append(0)
                            self.daten_sammler.alle_daten['ressourcennutzung']['gpu_speicher_verwendet_gb'].append(0)
                    except:
                        self.daten_sammler.alle_daten['ressourcennutzung']['gpu_prozent'].append(0)
                        self.daten_sammler.alle_daten['ressourcennutzung']['gpu_speicher_prozent'].append(0)
                        self.daten_sammler.alle_daten['ressourcennutzung']['gpu_speicher_verwendet_gb'].append(0)
                else:
                    self.daten_sammler.alle_daten['ressourcennutzung']['gpu_prozent'].append(0)
                    self.daten_sammler.alle_daten['ressourcennutzung']['gpu_speicher_prozent'].append(0)
                    self.daten_sammler.alle_daten['ressourcennutzung']['gpu_speicher_verwendet_gb'].append(0)
                
                time.sleep(1)  # Überwachung jede Sekunde
                
            except Exception as e:
                print(f"Überwachungsfehler: {e}")

class SkriptZeitmesser:
    def __init__(self, skript_name, daten_sammler):
        self.skript_name = skript_name
        self.daten_sammler = daten_sammler
        self.startzeit = None
        self.endzeit = None
        self.phasen_zeiten = {}
        self.aktuelle_phase = None
        self.phasen_start = None
        
    def starten(self):
        """Gesamtzeitmessung starten"""
        self.startzeit = time.time()
        print(f"{self.skript_name} gestartet um {datetime.now().strftime('%H:%M:%S')}")
        
    def phase_starten(self, phasen_name):
        """Neue Phase starten"""
        if self.aktuelle_phase:
            self.phase_beenden()
        self.aktuelle_phase = phasen_name
        self.phasen_start = time.time()
        
    def phase_beenden(self):
        """Aktuelle Phase beenden"""
        if self.aktuelle_phase and self.phasen_start:
            dauer = time.time() - self.phasen_start
            self.phasen_zeiten[self.aktuelle_phase] = dauer
            self.aktuelle_phase = None
            self.phasen_start = None
            
    def stoppen(self):
        """Gesamtzeitmessung stoppen"""
        if self.aktuelle_phase:
            self.phase_beenden()
        self.endzeit = time.time()
        gesamtzeit = self.endzeit - self.startzeit
        
        # Zeitmessungen in Datensammler speichern
        self.daten_sammler.alle_daten['zeitmessungen'] = {
            'gesamtlaufzeit_sekunden': gesamtzeit,
            'gesamtlaufzeit_minuten': gesamtzeit/60,
            **{f'phase_{k}_sekunden': v for k, v in self.phasen_zeiten.items()}
        }
        
        print(f"{self.skript_name} abgeschlossen in {gesamtzeit:.2f} Sekunden ({gesamtzeit/60:.1f} Minuten)")

# ==========================
# HILFSFUNKTIONEN
# ==========================
def unicode_imread(filepath):
    """Unicode-sichere Bildlade-Funktion für Windows"""
    try:
        # OpenCV mit Unicode-Support
        import numpy as np
        stream = open(filepath, "rb")
        bytes_data = bytearray(stream.read())
        numpyarray = np.asarray(bytes_data, dtype=np.uint8)
        image = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        stream.close()
        return image
    except:
        # Fallback zu normaler cv2.imread
        return cv2.imread(filepath)

def unicode_safe_path(path):
    """Konvertiert Pfad zu Unicode-sicherem Format"""
    try:
        return path.encode('utf-8').decode('utf-8')
    except:
        return path

# ==========================
# CUSTOM CNN MODELL
# ==========================
class VerkehrszeichenCNN(nn.Module):
    def __init__(self, anzahl_klassen, input_size=224):
        super(VerkehrszeichenCNN, self).__init__()
        self.input_size = input_size
        
        # Feature Extraction Layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.4),
        )
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, anzahl_klassen)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

# ==========================
# DATASET KLASSE
# ==========================
class GTSDBDataset(Dataset):
    def __init__(self, annotations, images_dir, transform=None, include_background=True, train_mode=True):
        self.annotations = annotations
        self.images_dir = images_dir
        self.transform = transform
        self.include_background = include_background
        self.train_mode = train_mode
        
        # Create samples list with proper class distribution
        self.samples = []
        
        # Group annotations by image
        image_annotations = {}
        for ann in annotations:
            img_name = ann['bildname']
            if img_name not in image_annotations:
                image_annotations[img_name] = []
            image_annotations[img_name].append(ann)
        
        # Add positive samples (with traffic signs)
        for img_name, img_anns in image_annotations.items():
            img_path = os.path.join(images_dir, img_name)
            if os.path.exists(img_path):
                for ann in img_anns:
                    self.samples.append({
                        'image_path': img_path,
                        'annotation': ann,
                        'has_sign': True
                    })
        
        # Add negative samples (background) if enabled and in training mode
        if include_background and train_mode:
            all_images = [f for f in os.listdir(images_dir) if f.endswith('.ppm')]
            annotated_images = set(image_annotations.keys())
            
            background_images = [img for img in all_images if img not in annotated_images]
            
            # Balance dataset: add background samples up to 30% of positive samples
            max_background = min(len(background_images), len(self.samples) // 3)
            
            for i, img_name in enumerate(background_images[:max_background]):
                img_path = os.path.join(images_dir, img_name)
                if os.path.exists(img_path):
                    self.samples.append({
                        'image_path': img_path,
                        'annotation': None,
                        'has_sign': False
                    })
        
        print(f"Dataset created with {len(self.samples)} samples")
        sign_samples = sum(1 for s in self.samples if s['has_sign'])
        bg_samples = len(self.samples) - sign_samples
        print(f"  - Traffic signs: {sign_samples}")
        print(f"  - Background: {bg_samples}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']
        
        # Unicode-sichere Bildladung
        image = unicode_imread(img_path)
        if image is None:
            # Return dummy data if image can't be loaded
            if self.transform:
                dummy_image = torch.zeros(3, 224, 224)
            else:
                dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            return dummy_image, 0  # Background class
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if not sample['has_sign']:
            # Background sample - use full image, resize and random crop
            if self.train_mode:
                # Random crop from full image for training
                h, w = image.shape[:2]
                crop_size = min(h, w, 100)  # Smaller crop for background
                if h > crop_size and w > crop_size:
                    start_y = random.randint(0, h - crop_size)
                    start_x = random.randint(0, w - crop_size)
                    image = image[start_y:start_y+crop_size, start_x:start_x+crop_size]
            
            label = 0  # Background class
        else:
            # Traffic sign sample
            ann = sample['annotation']
            x1, y1, x2, y2 = ann['links'], ann['oben'], ann['rechts'], ann['unten']
            
            # Add some padding around the bounding box
            h, w = image.shape[:2]
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Extract the sign region
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                # If crop is too small or empty, use full image
                cropped = image
            
            image = cropped
            # Map class_id to 1-based indexing (0 is background)
            label = ann['klassen_id'] + 1
        
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        
        return image, label

# ==========================
# TRAINING FUNCTIONS
# ==========================
def train_epoch(model, dataloader, criterion, optimizer, device, daten_sammler, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress tracking
    num_batches = len(dataloader)
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping für Stabilität
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % max(1, num_batches // 10) == 0:  # 10 Updates pro Epoche
            progress = 100.0 * batch_idx / num_batches
            print(f'Epoch {epoch}, Progress: {progress:.1f}%, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    # Calculate additional metrics
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss, epoch_acc, precision, recall, f1

# ==========================
# OPTIMALE BATCH-GRÖSSE BESTIMMUNG
# ==========================
def find_optimal_batch_size(model, sample_data, device, max_batch_size=128):
    """Optimale Batch-Größe für das System finden"""
    print("Bestimme optimale Batch-Größe...")
    
    model.eval()
    optimal_batch_size = 1
    
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        if batch_size > max_batch_size:
            break
            
        try:
            # Create dummy batch
            batch_data = sample_data.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
            
            # Test memory usage
            with torch.no_grad():
                start_time = time.time()
                _ = model(batch_data)
                end_time = time.time()
                
            # If successful, this batch size works
            optimal_batch_size = batch_size
            print(f"Batch-Größe {batch_size}: ✓ ({end_time - start_time:.3f}s)")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Batch-Größe {batch_size}: ✗ (Speicher-Fehler)")
                break
            else:
                print(f"Batch-Größe {batch_size}: ✗ ({str(e)})")
                break
        except Exception as e:
            print(f"Batch-Größe {batch_size}: ✗ ({str(e)})")
            break
    
    print(f"Optimale Batch-Größe: {optimal_batch_size}")
    return optimal_batch_size

# ==========================
# HAUPTSKRIPT
# ==========================
print("CUSTOM CNN VERKEHRSZEICHEN-PIPELINE - DATENSAMMLUNG")
print("="*60)

# Datensammler und Überwachung initialisieren
daten_sammler = DatenSammler()
zeitmesser = SkriptZeitmesser("Custom CNN Verkehrszeichen-Pipeline", daten_sammler)
monitor = RessourcenUeberwachung(daten_sammler)

# Überwachung starten
zeitmesser.starten()
monitor.ueberwachung_starten()

# ==========================
# 1. Setup und Gerätekonfiguration
# ==========================
zeitmesser.phase_starten("Setup und Konfiguration")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwendetes Gerät: {device}")

# ==========================
# 2. GTSDB Datensatz-Konfiguration
# ==========================
gtsdb_basis_pfad = r"C:\Users\timau\Desktop\Datensaetze\GTSDB"  # Ohne Umlaute!
gtsdb_train_pfad = os.path.join(gtsdb_basis_pfad, "Train") 
gtsdb_test_pfad = os.path.join(gtsdb_basis_pfad, "Test")
gt_train_datei = os.path.join(gtsdb_train_pfad, "gt-train.txt")
gt_test_datei = os.path.join(gtsdb_basis_pfad, "gt-test.txt")

# Ausgabeverzeichnis
ausgabe_verzeichnis = r"C:\Users\timau\Desktop"
os.makedirs(ausgabe_verzeichnis, exist_ok=True)

# Allgemeine Informationen sammeln
daten_sammler.alle_daten['allgemeine_info'] = {
    'datum': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'geraet': str(device),
    'seed': seed,
    'torch_version': torch.__version__,
    'cuda_verfuegbar': torch.cuda.is_available(),
    'gtsdb_train_pfad': gtsdb_train_pfad,
    'gtsdb_test_pfad': gtsdb_test_pfad,
    'modell_typ': 'Custom CNN'
}

zeitmesser.phase_beenden()

# ==========================
# 3. GTSDB Daten-Parser
# ==========================
zeitmesser.phase_starten("Datenanalyse")

def gtsdb_annotationen_parsen(gt_datei):
    """GTSDB Ground Truth Datei parsen"""
    annotationen = []
    with open(gt_datei, 'r') as f:
        for zeile in f:
            zeile = zeile.strip()
            if zeile and not zeile.startswith('#'):
                teile = zeile.split(';')
                if len(teile) == 6:
                    annotationen.append({
                        'bildname': teile[0],
                        'links': int(teile[1]),
                        'oben': int(teile[2]),
                        'rechts': int(teile[3]),
                        'unten': int(teile[4]),
                        'klassen_id': int(teile[5])
                    })
    return annotationen

# Annotationen parsen
print("Annotationen parsen...")
trainings_annotationen = gtsdb_annotationen_parsen(gt_train_datei)
test_annotationen = gtsdb_annotationen_parsen(gt_test_datei)

# Datensatzstatistiken
unique_classes = set([ann['klassen_id'] for ann in trainings_annotationen + test_annotationen])
num_classes = len(unique_classes) + 1  # +1 for background class

daten_sammler.alle_daten['allgemeine_info'].update({
    'anzahl_trainings_annotationen': len(trainings_annotationen),
    'anzahl_test_annotationen': len(test_annotationen),
    'anzahl_klassen': num_classes,
    'unique_klassen_ids': sorted(list(unique_classes))
})

print(f"Gefunden: {len(trainings_annotationen)} Training, {len(test_annotationen)} Test Annotationen")
print(f"Anzahl Klassen: {num_classes} (inkl. Hintergrund)")

zeitmesser.phase_beenden()

# ==========================
# 4. Datenloaders erstellen
# ==========================
zeitmesser.phase_starten("Datenloader Erstellung")

# Data transforms mit verbesserter Augmentation - KEINE SPIEGELUNGEN!
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Größer für bessere Qualität
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Besseres Cropping
    # KEINE RandomHorizontalFlip oder VerticalFlip - Verkehrszeichen sind orientierungsabhängig!
    transforms.RandomRotation(degrees=3),     # Minimal Rotation (nur 3°)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),  # Reduzierte Farbänderungen
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),  # Minimale Translation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets erstellen mit verbesserter Konfiguration
print("Datasets erstellen...")
train_dataset = GTSDBDataset(
    trainings_annotationen, 
    gtsdb_train_pfad, 
    transform=train_transform, 
    include_background=True,
    train_mode=True
)
test_dataset = GTSDBDataset(
    test_annotationen, 
    gtsdb_test_pfad, 
    transform=val_transform,
    include_background=False,  # Kein Background für Validation
    train_mode=False
)

daten_sammler.alle_daten['allgemeine_info'].update({
    'training_dataset_groesse': len(train_dataset),
    'test_dataset_groesse': len(test_dataset)
})

zeitmesser.phase_beenden()

# ==========================
# 5. Modell initialisieren und optimale Batch-Größe finden
# ==========================
zeitmesser.phase_starten("Modell Initialisierung")

print("Modell initialisieren...")
model = VerkehrszeichenCNN(anzahl_klassen=num_classes).to(device)

# Sample data für Batch-Size-Optimierung
sample_data = torch.randn(1, 3, 224, 224)
optimal_batch_size = find_optimal_batch_size(model, sample_data, device, max_batch_size=64)

# DataLoaders mit optimaler Batch-Größe erstellen
train_loader = DataLoader(
    train_dataset, 
    batch_size=optimal_batch_size, 
    shuffle=True, 
    num_workers=0,  # Windows-kompatibel
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=optimal_batch_size, 
    shuffle=False, 
    num_workers=0,  # Windows-kompatibel
    pin_memory=torch.cuda.is_available()
)

daten_sammler.alle_daten['allgemeine_info'].update({
    'optimale_batch_groesse': optimal_batch_size,
    'modell_parameter': sum(p.numel() for p in model.parameters()),
    'trainierbare_parameter': sum(p.numel() for p in model.parameters() if p.requires_grad)
})

print(f"Modell Parameter: {sum(p.numel() for p in model.parameters()):,}")

zeitmesser.phase_beenden()

# ==========================
# 6. Training Setup mit verbesserter Konfiguration
# ==========================
zeitmesser.phase_starten("Training")

# Klassenwichtungen berechnen für unbalancierte Daten
def calculate_class_weights(dataset):
    """Berechne Klassenwichtungen für unbalancierte Daten"""
    class_counts = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        label = label.item() if torch.is_tensor(label) else label
        class_counts[label] = class_counts.get(label, 0) + 1
    
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    weights = {}
    for class_id, count in class_counts.items():
        weights[class_id] = total_samples / (num_classes * count)
    
    # Sortiere nach Klassen-ID
    weight_list = [weights.get(i, 1.0) for i in range(max(weights.keys()) + 1)]
    return torch.FloatTensor(weight_list)

# Klassenwichtungen berechnen
print("Berechne Klassenwichtungen...")
class_weights = calculate_class_weights(train_dataset).to(device)
print(f"Klassenwichtungen: {class_weights[:10]}...")  # Zeige erste 10

# Loss function mit Klassenwichtungen und verbesserter Konfiguration
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# Verbesserte Optimizer-Konfiguration
optimizer = optim.AdamW(
    model.parameters(), 
    lr=0.0005,  # Niedrigere Learning Rate
    weight_decay=1e-3,  # Stärkere Regularisierung
    betas=(0.9, 0.999)
)

# Verbesserter Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10,  # Restart alle 10 Epochen
    T_mult=1, 
    eta_min=1e-6
)

# Training Parameter
num_epochs = 50  # Mehr Epochen
best_val_acc = 0.0
patience = 10    # Mehr Geduld
patience_counter = 0

print(f"Training starten - {num_epochs} Epochen mit Batch-Größe {optimal_batch_size}")
print(f"Klassenwichtungen aktiviert, Label Smoothing: 0.1")

# Training Loop
for epoch in range(num_epochs):
    print(f"\nEpoche {epoch+1}/{num_epochs}")
    print("-" * 30)
    
    # Training
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, daten_sammler, epoch+1)
    
    # Validation
    val_loss, val_acc, precision, recall, f1 = validate_epoch(model, test_loader, criterion, device)
    
    # Learning rate scheduler
    scheduler.step()
    
    # Metriken sammeln
    daten_sammler.alle_daten['training_metriken']['epochen'].append(epoch + 1)
    daten_sammler.alle_daten['training_metriken']['train_loss'].append(train_loss)
    daten_sammler.alle_daten['training_metriken']['train_accuracy'].append(train_acc)
    daten_sammler.alle_daten['training_metriken']['val_loss'].append(val_loss)
    daten_sammler.alle_daten['training_metriken']['val_accuracy'].append(val_acc)
    daten_sammler.alle_daten['training_metriken']['precision'].append(precision)
    daten_sammler.alle_daten['training_metriken']['recall'].append(recall)
    daten_sammler.alle_daten['training_metriken']['f1_score'].append(f1)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Best model speichern
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        best_model_path = os.path.join(ausgabe_verzeichnis, "best_traffic_sign_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc,
            'num_classes': num_classes,
            'class_weights': class_weights
        }, best_model_path)
        print(f"★ Neues bestes Modell gespeichert! Validation Accuracy: {val_acc:.2f}%")
    else:
        patience_counter += 1
        print(f"Keine Verbesserung seit {patience_counter} Epochen")
        
    # Early stopping
    if patience_counter >= patience:
        print(f"Early stopping nach {epoch+1} Epochen (keine Verbesserung seit {patience} Epochen)")
        break

# Finales Modell speichern
final_model_path = os.path.join(ausgabe_verzeichnis, "final_traffic_sign_model.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'val_acc': val_acc,
    'num_classes': num_classes
}, final_model_path)

daten_sammler.alle_daten['allgemeine_info'].update({
    'best_model_path': best_model_path,
    'final_model_path': final_model_path,
    'best_validation_accuracy': best_val_acc,
    'training_epochen_abgeschlossen': epoch + 1,
    'early_stopping_verwendet': patience_counter >= patience
})

print("Training abgeschlossen!")

zeitmesser.phase_beenden()

# ==========================
# 7. Modell für Inferenz laden
# ==========================
zeitmesser.phase_starten("Modell Laden für Inferenz")

print("Bestes Modell für Inferenz laden...")

# Bestes Modell laden
checkpoint = torch.load(best_model_path, map_location=device)
inference_model = VerkehrszeichenCNN(anzahl_klassen=num_classes).to(device)
inference_model.load_state_dict(checkpoint['model_state_dict'])
inference_model.eval()

# Klassen-Mapping erstellen (vereinfacht)
class_names = {0: 'Hintergrund/Kein_Zeichen'}
for i in range(1, num_classes):
    class_names[i] = f'Verkehrszeichen_Klasse_{i-1}'

daten_sammler.alle_daten['allgemeine_info']['klassen_mapping'] = class_names

zeitmesser.phase_beenden()

# ==========================
# 8. Pipeline-Klasse für Inferenz
# ==========================
zeitmesser.phase_starten("Pipeline-Tests")

class VerkehrszeichenErkennungsPipeline:
    def __init__(self, model, transform, class_names, device):
        self.model = model
        self.transform = transform
        self.class_names = class_names
        self.device = device
        
    def predict_image(self, image_path, return_confidence=True):
        """Einzelbild klassifizieren"""
        try:
            # Unicode-sichere Bildladung
            image = unicode_imread(image_path)
            if image is None:
                return None, 0.0
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)
            
            # Transform anwenden
            input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
            
            # Inferenz
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = predicted.item()
                confidence_score = confidence.item()
            
            if return_confidence:
                return predicted_class, confidence_score
            else:
                return predicted_class
                
        except Exception as e:
            print(f"Fehler bei der Vorhersage für {image_path}: {e}")
            return None, 0.0
    
    def predict_batch(self, image_paths, batch_size=None):
        """Batch von Bildern klassifizieren"""
        if batch_size is None:
            batch_size = len(image_paths)
        
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_indices = []
            
            # Batch vorbereiten
            for j, path in enumerate(batch_paths):
                try:
                    image = unicode_imread(path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image_pil = Image.fromarray(image)
                        transformed = self.transform(image_pil)
                        batch_images.append(transformed)
                        valid_indices.append(j)
                except Exception as e:
                    continue
            
            if not batch_images:
                # Fallback für leere Batches
                for path in batch_paths:
                    results.append((None, 0.0))
                continue
            
            # Batch-Inferenz
            try:
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidences, predictions = torch.max(probabilities, 1)
                
                # Ergebnisse zuordnen
                batch_results = [None] * len(batch_paths)
                for k, valid_idx in enumerate(valid_indices):
                    pred_class = predictions[k].item()
                    conf_score = confidences[k].item()
                    batch_results[valid_idx] = (pred_class, conf_score)
                
                # Fehlende Ergebnisse mit None füllen
                for k in range(len(batch_paths)):
                    if batch_results[k] is None:
                        batch_results[k] = (None, 0.0)
                
                results.extend(batch_results)
                
            except Exception as e:
                print(f"Batch-Fehler: {e}")
                for path in batch_paths:
                    results.append((None, 0.0))
        
        return results

# Pipeline initialisieren
print("Pipeline initialisieren...")
pipeline = VerkehrszeichenErkennungsPipeline(
    model=inference_model,
    transform=val_transform,
    class_names=class_names,
    device=device
)

print("Pipeline-Tests durchführen...")

# Testbilder sammeln
test_bilder = []
for bild_datei in os.listdir(gtsdb_test_pfad):
    if bild_datei.endswith('.ppm'):
        test_bilder.append(os.path.join(gtsdb_test_pfad, bild_datei))

anzahl_test_bilder = min(20, len(test_bilder))

# Einzelbild-Tests
for i, test_bild_pfad in enumerate(test_bilder[:anzahl_test_bilder]):
    start_zeit = time.time()
    predicted_class, confidence = pipeline.predict_image(test_bild_pfad)
    end_zeit = time.time()
    
    verarbeitungszeit = end_zeit - start_zeit
    
    if predicted_class is not None:
        # Inferenzzeit sammeln
        daten_sammler.alle_daten['leistungsmetriken']['inferenzzeiten'].append(verarbeitungszeit * 1000)
        daten_sammler.alle_daten['leistungsmetriken']['erkennungskonfidenzen'].append(confidence)
        daten_sammler.alle_daten['leistungsmetriken']['klassifikationskonfidenzen'].append(confidence)
        
        # Einzelbild-Ergebnis sammeln
        einzelbild_ergebnis = {
            'bild_nr': i + 1,
            'bildname': os.path.basename(test_bild_pfad),
            'verarbeitungszeit_ms': verarbeitungszeit * 1000,
            'vorhergesagte_klasse': predicted_class,
            'klassen_name': class_names.get(predicted_class, 'Unbekannt'),
            'konfidenz': confidence,
            'ist_verkehrszeichen': predicted_class > 0
        }
        
        daten_sammler.alle_daten['einzelbild_ergebnisse'].append(einzelbild_ergebnis)

print(f"Pipeline-Tests abgeschlossen - {anzahl_test_bilder} Bilder verarbeitet")

zeitmesser.phase_beenden()

# ==========================
# 9. Leistungsevaluierung und Benchmarks
# ==========================
zeitmesser.phase_starten("Leistungsevaluierung")

print("Leistungsbenchmarks durchführen...")

def pipeline_benchmarken(pipeline, test_images, batch_groesse):
    """Pipeline mit verschiedenen Batch-Größen benchmarken"""
    if not test_images:
        return {}
    
    # Begrenzte Anzahl für Benchmark
    benchmark_images = test_images[:min(50, len(test_images))]
    
    # Aufwärmphase - Einzelbilder
    for _ in range(5):
        pipeline.predict_image(benchmark_images[0])
    
    # Einzelbild-Benchmark
    single_times = []
    for _ in range(20):
        start_time = time.perf_counter()
        pipeline.predict_image(benchmark_images[0])
        end_time = time.perf_counter()
        single_times.append(end_time - start_time)
    
    # Batch-Benchmark
    if batch_groesse > 1 and len(benchmark_images) >= batch_groesse:
        batch_images = benchmark_images[:batch_groesse]
        batch_times = []
        
        for _ in range(10):
            start_time = time.perf_counter()
            pipeline.predict_batch(batch_images, batch_size=batch_groesse)
            end_time = time.perf_counter()
            batch_times.append((end_time - start_time) / batch_groesse)  # Pro Bild
    else:
        batch_times = single_times
    
    # Statistiken berechnen
    single_latenz = np.mean(single_times) * 1000  # ms
    batch_latenz = np.mean(batch_times) * 1000    # ms
    
    return {
        'einzelbild_latenz_ms': single_latenz,
        'batch_latenz_pro_bild_ms': batch_latenz,
        'einzelbild_durchsatz_bilder_pro_sekunde': 1000 / single_latenz,
        'batch_durchsatz_bilder_pro_sekunde': 1000 / batch_latenz,
        'speedup_faktor': single_latenz / batch_latenz,
        'min_latenz_ms': np.min(single_times) * 1000,
        'max_latenz_ms': np.max(single_times) * 1000,
        'p95_latenz_ms': np.percentile(single_times, 95) * 1000,
        'p99_latenz_ms': np.percentile(single_times, 99) * 1000,
        'standardabweichung_ms': np.std(single_times) * 1000
    }

# Benchmarks für verschiedene Batch-Größen
batch_groessen = [1, 2, 4, 8, 16, optimal_batch_size]

for batch_groesse in batch_groessen:
    if batch_groesse <= optimal_batch_size:
        benchmark_ergebnis = pipeline_benchmarken(pipeline, test_bilder, batch_groesse)
        if benchmark_ergebnis:
            daten_sammler.alle_daten['leistungsmetriken']['batch_benchmarks'][f'batch_{batch_groesse}'] = benchmark_ergebnis
            print(f"Batch-Größe {batch_groesse}: {benchmark_ergebnis['einzelbild_latenz_ms']:.2f}ms")

print("Benchmarks abgeschlossen")

zeitmesser.phase_beenden()

# ==========================
# 10. Erweiterte Pipeline-Evaluierung
# ==========================
zeitmesser.phase_starten("Erweiterte Evaluierung")

print("Erweiterte Pipeline-Evaluierung durchführen...")

# Test mit verschiedenen Konfidenz-Schwellwerten
konfidenz_schwellwerte = [0.1, 0.3, 0.5, 0.7, 0.9]
pipeline_test_zusammenfassung = []

for schwellwert in konfidenz_schwellwerte:
    erkennungen_verkehrszeichen = 0
    erkennungen_hintergrund = 0
    verarbeitungszeiten = []
    konfidenzen = []
    
    test_anzahl = min(30, len(test_bilder))
    
    for test_bild_pfad in test_bilder[:test_anzahl]:
        start_zeit = time.time()
        predicted_class, confidence = pipeline.predict_image(test_bild_pfad)
        end_zeit = time.time()
        
        if predicted_class is not None and confidence >= schwellwert:
            verarbeitungszeiten.append((end_zeit - start_zeit) * 1000)
            konfidenzen.append(confidence)
            
            if predicted_class > 0:
                erkennungen_verkehrszeichen += 1
            else:
                erkennungen_hintergrund += 1
    
    # Zusammenfassung für diesen Schwellwert
    test_zusammenfassung = {
        'konfidenz_schwellwert': schwellwert,
        'durchschnittliche_verarbeitungszeit_ms': np.mean(verarbeitungszeiten) if verarbeitungszeiten else 0,
        'verkehrszeichen_erkennungen': erkennungen_verkehrszeichen,
        'hintergrund_erkennungen': erkennungen_hintergrund,
        'gesamt_erkennungen': erkennungen_verkehrszeichen + erkennungen_hintergrund,
        'verkehrszeichen_rate': erkennungen_verkehrszeichen / test_anzahl * 100,
        'durchschnittliche_konfidenz': np.mean(konfidenzen) if konfidenzen else 0,
        'median_verarbeitungszeit_ms': np.median(verarbeitungszeiten) if verarbeitungszeiten else 0,
        'p95_verarbeitungszeit_ms': np.percentile(verarbeitungszeiten, 95) if verarbeitungszeiten else 0
    }
    
    pipeline_test_zusammenfassung.append(test_zusammenfassung)

# Pipeline-Tests in Datensammler speichern
daten_sammler.alle_daten['pipeline_testmetriken'] = pipeline_test_zusammenfassung

print("Erweiterte Evaluierung abgeschlossen")

zeitmesser.phase_beenden()

# ==========================
# 11. Abschluss und Datenexport
# ==========================
zeitmesser.phase_starten("Datenexport")

# Überwachung stoppen
monitor.ueberwachung_stoppen()
zeitmesser.stoppen()

# Zusammenfassungsstatistiken berechnen
if daten_sammler.alle_daten['leistungsmetriken']['inferenzzeiten']:
    inferenzzeiten = daten_sammler.alle_daten['leistungsmetriken']['inferenzzeiten']
    konfidenzen = daten_sammler.alle_daten['leistungsmetriken']['erkennungskonfidenzen']
    
    zusammenfassung_statistiken = {
        'durchschnittliche_inferenzzeit_ms': np.mean(inferenzzeiten),
        'median_inferenzzeit_ms': np.median(inferenzzeiten),
        'p95_inferenzzeit_ms': np.percentile(inferenzzeiten, 95),
        'p99_inferenzzeit_ms': np.percentile(inferenzzeiten, 99),
        'standardabweichung_inferenzzeit_ms': np.std(inferenzzeiten),
        'min_inferenzzeit_ms': np.min(inferenzzeiten),
        'max_inferenzzeit_ms': np.max(inferenzzeiten),
        'durchschnittlicher_durchsatz_bilder_pro_sekunde': 1000 / np.mean(inferenzzeiten),
        'durchschnittliche_konfidenz': np.mean(konfidenzen) if konfidenzen else 0,
        'median_konfidenz': np.median(konfidenzen) if konfidenzen else 0,
        'gesamt_verarbeitete_bilder': len(inferenzzeiten),
        'verkehrszeichen_erkennungen': sum(1 for r in daten_sammler.alle_daten['einzelbild_ergebnisse'] if r.get('ist_verkehrszeichen', False))
    }
    
    daten_sammler.alle_daten['allgemeine_info'].update(zusammenfassung_statistiken)

# Ressourcenzusammenfassung
if daten_sammler.alle_daten['ressourcennutzung']['zeitstempel']:
    ressourcen_zusammenfassung = {
        'durchschnittliche_cpu_prozent': np.mean(daten_sammler.alle_daten['ressourcennutzung']['cpu_prozent']),
        'maximale_cpu_prozent': np.max(daten_sammler.alle_daten['ressourcennutzung']['cpu_prozent']),
        'durchschnittliches_ram_gb': np.mean(daten_sammler.alle_daten['ressourcennutzung']['ram_verwendet_gb']),
        'maximales_ram_gb': np.max(daten_sammler.alle_daten['ressourcennutzung']['ram_verwendet_gb']),
        'durchschnittliche_gpu_prozent': np.mean(daten_sammler.alle_daten['ressourcennutzung']['gpu_prozent']),
        'maximale_gpu_prozent': np.max(daten_sammler.alle_daten['ressourcennutzung']['gpu_prozent']),
        'durchschnittliches_vram_gb': np.mean(daten_sammler.alle_daten['ressourcennutzung']['gpu_speicher_verwendet_gb']),
        'maximales_vram_gb': np.max(daten_sammler.alle_daten['ressourcennutzung']['gpu_speicher_verwendet_gb'])
    }
    daten_sammler.alle_daten['allgemeine_info'].update(ressourcen_zusammenfassung)

# Training-Zusammenfassung
if daten_sammler.alle_daten['training_metriken']['val_accuracy']:
    training_zusammenfassung = {
        'finale_train_accuracy': daten_sammler.alle_daten['training_metriken']['train_accuracy'][-1],
        'finale_val_accuracy': daten_sammler.alle_daten['training_metriken']['val_accuracy'][-1],
        'beste_val_accuracy': max(daten_sammler.alle_daten['training_metriken']['val_accuracy']),
        'finale_precision': daten_sammler.alle_daten['training_metriken']['precision'][-1],
        'finale_recall': daten_sammler.alle_daten['training_metriken']['recall'][-1],
        'finale_f1_score': daten_sammler.alle_daten['training_metriken']['f1_score'][-1]
    }
    daten_sammler.alle_daten['allgemeine_info'].update(training_zusammenfassung)

# Leistungskategorie bestimmen
if daten_sammler.alle_daten['leistungsmetriken']['inferenzzeiten']:
    avg_inferenz = np.mean(daten_sammler.alle_daten['leistungsmetriken']['inferenzzeiten'])
    if avg_inferenz < 50:
        leistungskategorie = "Hochperformant (< 50ms)"
    elif avg_inferenz < 100:
        leistungskategorie = "Echtzeitfähig (< 100ms)"
    elif avg_inferenz < 500:
        leistungskategorie = "Nahezu Echtzeit (< 500ms)"
    else:
        leistungskategorie = "Batch-Verarbeitung geeignet (> 500ms)"
    
    daten_sammler.alle_daten['allgemeine_info']['leistungskategorie'] = leistungskategorie

# Excel-Datei erstellen
excel_pfad = os.path.join(ausgabe_verzeichnis, "custom_cnn_pipeline_ergebnisse.xlsx")
daten_sammler.zu_excel_exportieren(excel_pfad)

zeitmesser.phase_beenden()

# ==========================
# ABSCHLUSSBERICHT
# ==========================
print("\n" + "="*80)
print("CUSTOM CNN PIPELINE ABGESCHLOSSEN")
print("="*80)
print(f"Gesamtlaufzeit: {daten_sammler.alle_daten['zeitmessungen']['gesamtlaufzeit_minuten']:.1f} Minuten")
print()
print("TRAININGSERGEBNISSE:")
if daten_sammler.alle_daten['training_metriken']['val_accuracy']:
    beste_acc = max(daten_sammler.alle_daten['training_metriken']['val_accuracy'])
    finale_acc = daten_sammler.alle_daten['training_metriken']['val_accuracy'][-1]
    finale_f1 = daten_sammler.alle_daten['training_metriken']['f1_score'][-1]
    
    print(f"Beste Validation Accuracy: {beste_acc:.2f}%")
    print(f"Finale Validation Accuracy: {finale_acc:.2f}%")
    print(f"Finale F1-Score: {finale_f1:.4f}")
    print(f"Trainierte Epochen: {len(daten_sammler.alle_daten['training_metriken']['epochen'])}")

print()
print("INFERENZ-LEISTUNG:")
if daten_sammler.alle_daten['leistungsmetriken']['inferenzzeiten']:
    avg_inferenz = np.mean(daten_sammler.alle_daten['leistungsmetriken']['inferenzzeiten'])
    avg_durchsatz = 1000 / avg_inferenz
    p95_latenz = np.percentile(daten_sammler.alle_daten['leistungsmetriken']['inferenzzeiten'], 95)
    avg_konfidenz = np.mean(daten_sammler.alle_daten['leistungsmetriken']['erkennungskonfidenzen'])
    
    print(f"Inferenzgeschwindigkeit: {avg_inferenz:.2f} ms (Durchschnitt)")
    print(f"Latenz P95: {p95_latenz:.2f} ms")
    print(f"Durchsatz: {avg_durchsatz:.1f} Bilder/Sekunde")
    print(f"Durchschnittliche Konfidenz: {avg_konfidenz:.3f}")
    print(f"Optimale Batch-Größe: {optimal_batch_size}")

# Leistungsbewertung
if 'leistungskategorie' in daten_sammler.alle_daten['allgemeine_info']:
    print(f"Leistungsbewertung: {daten_sammler.alle_daten['allgemeine_info']['leistungskategorie']}")

print()
print("GESPEICHERTE DATEIEN:")
print(f"- Bestes Modell: {os.path.basename(best_model_path)}")
print(f"- Finales Modell: {os.path.basename(final_model_path)}")
print(f"- Excel-Daten: {os.path.basename(excel_pfad)}")
print()
print("EXCEL-ARBEITSBLÄTTER:")
print("- Allgemeine_Info: Systeminfo, Modellarchitektur, Leistungszusammenfassung")
print("- Training_Metriken: Epochen-weise Trainingsmetriken (Accuracy, Loss, F1-Score)")
print("- Pipeline_Tests: Tests mit verschiedenen Konfidenz-Schwellwerten")
print("- Einzelbild_Ergebnisse: Detaillierte Klassifikationsergebnisse pro Bild")
print("- Inferenz_Zeiten: Inferenzzeiten und Konfidenzwerte aller Tests")
print("- Batch_Benchmarks: Leistungsmetriken für verschiedene Batch-Größen")
print("- Ressourcennutzung: CPU/RAM/GPU-Verbrauch über Zeit")
print("- Zeitmessungen: Ausführungszeiten aller Phasen")
print()
print("MODELL-DETAILS:")
print(f"- Architektur: Custom CNN mit {sum(p.numel() for p in model.parameters()):,} Parametern")
print(f"- Eingabegröße: 224x224 RGB")
print(f"- Ausgabe-Klassen: {num_classes} (inkl. Hintergrund)")
print(f"- Optimierter Batch-Size: {optimal_batch_size}")
print()
print("NÄCHSTE SCHRITTE:")
print("→ Verwenden Sie das beste Modell für Deployment")
print("→ Analysieren Sie die Trainingsmetriken für weitere Optimierungen")
print("→ Nutzen Sie die Batch-Benchmarks für Produktions-Setup")
print("→ Überprüfen Sie Pipeline-Tests für optimale Konfidenz-Schwellwerte")
print("="*80)