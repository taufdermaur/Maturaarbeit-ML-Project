import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import numpy as np
import random
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import torch.nn.functional as F
import sys
from io import StringIO

# ==========================
# MONITORING IMPORTS
# ==========================
import psutil
import threading
import json
from datetime import datetime
import time
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPUtil not available - GPU monitoring disabled")

# ==========================
# MONITORING CLASSES
# ==========================
class ResourceMonitor:
    def __init__(self):
        self.monitoring = False
        self.data = {
            'timestamps': [],
            'cpu_percent': [],
            'ram_percent': [],
            'ram_used_gb': [],
            'gpu_percent': [],
            'gpu_memory_percent': [],
            'gpu_memory_used_gb': []
        }
        self.start_time = None
        self.end_time = None
        
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        print(f"Resource monitoring started at {datetime.now().strftime('%H:%M:%S')}")
        
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        self.end_time = time.time()
        print(f"Resource monitoring stopped at {datetime.now().strftime('%H:%M:%S')}")
        
    def _monitor_resources(self):
        """Monitoring loop (runs in separate thread)"""
        while self.monitoring:
            try:
                current_time = time.time() - self.start_time
                self.data['timestamps'].append(current_time)
                
                # CPU monitoring
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.data['cpu_percent'].append(cpu_percent)
                
                # RAM monitoring
                ram = psutil.virtual_memory()
                self.data['ram_percent'].append(ram.percent)
                self.data['ram_used_gb'].append(ram.used / (1024**3))
                
                # GPU monitoring
                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            self.data['gpu_percent'].append(gpu.load * 100)
                            self.data['gpu_memory_percent'].append(gpu.memoryUtil * 100)
                            self.data['gpu_memory_used_gb'].append(gpu.memoryUsed / 1024)
                        else:
                            self.data['gpu_percent'].append(0)
                            self.data['gpu_memory_percent'].append(0)
                            self.data['gpu_memory_used_gb'].append(0)
                    except:
                        self.data['gpu_percent'].append(0)
                        self.data['gpu_memory_percent'].append(0)
                        self.data['gpu_memory_used_gb'].append(0)
                else:
                    self.data['gpu_percent'].append(0)
                    self.data['gpu_memory_percent'].append(0)
                    self.data['gpu_memory_used_gb'].append(0)
                
                time.sleep(2)  # Monitor every 2 seconds for training
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                
    def get_summary(self):
        """Create summary of resource usage"""
        if not self.data['timestamps']:
            return "No monitoring data available"
            
        total_time = self.end_time - self.start_time if self.end_time else time.time() - self.start_time
        
        summary = f"""
RESOURCE MONITORING SUMMARY
{'='*60}
Total runtime: {total_time:.2f} seconds ({total_time/60:.1f} minutes)

CPU:
   - Average: {np.mean(self.data['cpu_percent']):.1f}%
   - Maximum: {np.max(self.data['cpu_percent']):.1f}%
   - Minimum: {np.min(self.data['cpu_percent']):.1f}%

RAM:
   - Average usage: {np.mean(self.data['ram_percent']):.1f}%
   - Maximum usage: {np.max(self.data['ram_percent']):.1f}%
   - Average consumption: {np.mean(self.data['ram_used_gb']):.2f} GB
   - Maximum consumption: {np.max(self.data['ram_used_gb']):.2f} GB

GPU:
   - Average usage: {np.mean(self.data['gpu_percent']):.1f}%
   - Maximum usage: {np.max(self.data['gpu_percent']):.1f}%
   - Average VRAM consumption: {np.mean(self.data['gpu_memory_used_gb']):.2f} GB
   - Maximum VRAM consumption: {np.max(self.data['gpu_memory_used_gb']):.2f} GB
{'='*60}
"""
        return summary
    
    def plot_resources(self, save_path=None):
        """Create plots of resource usage"""
        if not self.data['timestamps']:
            print("No data available for plotting")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Resource Usage During CNN Training', fontsize=16)
        
        # CPU plot
        axes[0,0].plot(self.data['timestamps'], self.data['cpu_percent'], 'b-', linewidth=2)
        axes[0,0].set_title('CPU Usage')
        axes[0,0].set_xlabel('Time (seconds)')
        axes[0,0].set_ylabel('CPU %')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_ylim(0, 100)
        
        # RAM plot
        axes[0,1].plot(self.data['timestamps'], self.data['ram_used_gb'], 'g-', linewidth=2)
        axes[0,1].set_title('RAM Usage')
        axes[0,1].set_xlabel('Time (seconds)')
        axes[0,1].set_ylabel('RAM (GB)')
        axes[0,1].grid(True, alpha=0.3)
        
        # GPU usage plot
        axes[1,0].plot(self.data['timestamps'], self.data['gpu_percent'], 'r-', linewidth=2)
        axes[1,0].set_title('GPU Usage')
        axes[1,0].set_xlabel('Time (seconds)')
        axes[1,0].set_ylabel('GPU %')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_ylim(0, 100)
        
        # GPU memory plot
        axes[1,1].plot(self.data['timestamps'], self.data['gpu_memory_used_gb'], 'm-', linewidth=2)
        axes[1,1].set_title('GPU Memory Usage')
        axes[1,1].set_xlabel('Time (seconds)')
        axes[1,1].set_ylabel('VRAM (GB)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Resource plot saved: {save_path}")
        
        plt.show()

class ScriptTimer:
    def __init__(self, script_name):
        self.script_name = script_name
        self.start_time = None
        self.end_time = None
        self.phase_times = {}
        self.current_phase = None
        self.phase_start = None
        
    def start(self):
        """Start overall timing"""
        self.start_time = time.time()
        print(f"{self.script_name} started at {datetime.now().strftime('%H:%M:%S')}")
        
    def start_phase(self, phase_name):
        """Start a new phase"""
        if self.current_phase:
            self.end_phase()
        self.current_phase = phase_name
        self.phase_start = time.time()
        print(f"Phase '{phase_name}' started")
        
    def end_phase(self):
        """End current phase"""
        if self.current_phase and self.phase_start:
            duration = time.time() - self.phase_start
            self.phase_times[self.current_phase] = duration
            print(f"Phase '{self.current_phase}' completed in {duration:.2f} seconds")
            self.current_phase = None
            self.phase_start = None
            
    def stop(self):
        """Stop overall timing"""
        if self.current_phase:
            self.end_phase()
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        print(f"{self.script_name} completed at {datetime.now().strftime('%H:%M:%S')}")
        print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        
    def get_summary(self):
        """Return timing summary"""
        if not self.start_time or not self.end_time:
            return "Timing measurement incomplete"
            
        total_time = self.end_time - self.start_time
        
        summary = f"""
TIMING SUMMARY for {self.script_name}
{'='*60}
Total runtime: {total_time:.2f} seconds ({total_time/60:.1f} minutes)

Phase breakdown:
"""
        
        for phase, duration in self.phase_times.items():
            percentage = (duration / total_time) * 100
            summary += f"   - {phase}: {duration:.2f}s ({percentage:.1f}%)\n"
            
        summary += f"{'='*60}"
        return summary

# ==========================
# MONITORING INITIALIZATION
# ==========================
print("CNN TRAINING SCRIPT WITH MONITORING")
print("="*60)

# Initialize timer and monitor
timer = ScriptTimer("CNN Training for ASTRA and GTSRB")
monitor = ResourceMonitor()

# Start monitoring
timer.start()
monitor.start_monitoring()

# ==========================
# 1. Reproducibility
# ==========================
timer.start_phase("Setup and Initialization")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# ==========================
# 2. Device
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================
# 3. Transform
# ==========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

timer.end_phase()

# ==========================
# 4. Custom Dataset for GTSRB Testing
# ==========================
timer.start_phase("Dataset Preparation")

class GTSRBTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, train_classes=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.train_classes = train_classes
        
        if train_classes is not None:
            self.label_mapping = {}
            for folder_name in train_classes:
                class_id = int(folder_name.split('_')[0])
                train_label = train_classes.index(folder_name)
                self.label_mapping[class_id] = train_label
        else:
            self.label_mapping = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['Path'])
        image = Image.open(img_path).convert('RGB')
        csv_class_id = self.data.iloc[idx]['ClassId']
        
        if self.label_mapping is not None and csv_class_id in self.label_mapping:
            label = self.label_mapping[csv_class_id]
        else:
            label = csv_class_id
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==========================
# 5. Load datasets separately
# ==========================
astra_train_path = r"C:\Users\timau\Desktop\Datensätze\ASTRA\Train"
astra_test_path  = r"C:\Users\timau\Desktop\Datensätze\ASTRA\Test"
gtsrb_train_path = r"C:\Users\timau\Desktop\Datensätze\GTSRB\Train"
gtsrb_test_csv   = r"C:\Users\timau\Desktop\Datensätze\GTSRB\Test.csv"
gtsrb_base_path  = r"C:\Users\timau\Desktop\Datensätze\GTSRB"

astra_train_full = datasets.ImageFolder(root=astra_train_path, transform=transform)
astra_test       = datasets.ImageFolder(root=astra_test_path, transform=transform)

gtsrb_train_full = datasets.ImageFolder(root=gtsrb_train_path, transform=transform)

gtsrb_test = GTSRBTestDataset(csv_file=gtsrb_test_csv, root_dir=gtsrb_base_path, 
                              transform=transform, train_classes=gtsrb_train_full.classes)

# ==========================
# 6. Split train sets → train + val
# ==========================
val_ratio = 0.2

# ASTRA Split
astra_val_size = int(len(astra_train_full) * val_ratio)
astra_train_size = len(astra_train_full) - astra_val_size
astra_train, astra_val = random_split(astra_train_full, [astra_train_size, astra_val_size], 
                                      generator=torch.Generator().manual_seed(seed))

# GTSRB Split
gtsrb_val_size = int(len(gtsrb_train_full) * val_ratio)
gtsrb_train_size = len(gtsrb_train_full) - gtsrb_val_size
gtsrb_train, gtsrb_val = random_split(gtsrb_train_full, [gtsrb_train_size, gtsrb_val_size], 
                                      generator=torch.Generator().manual_seed(seed))

astra_num_classes = len(astra_train_full.classes)
gtsrb_num_classes = len(gtsrb_train_full.classes)

# Class names for ASTRA
astra_class_names = [
    '0_Besondere_Signale',
    '1_Ergaenzende_Angaben_zu_Signalen',
    '2_Fahranordnungen_Parkierungsbeschraenkungen',
    '3_Fahrverbote_Mass_und_Gewichtsbeschraenkungen',
    '4_Informationshinweise',
    '5_Markierungen_und_Leiteinrichtugen',
    '6_Verhaltenshinweise',
    '7_Vortrittssignale',
    '8_Wegweisung_aufAutobahnen_und_Autostrassen',
    '9_Wegweisung_auf_Haupt_und_Nebenstrassen'
]

# Class names for GTSRB
gtsrb_class_names = [
    '0_Geschwindigkeitsbeschraenkung_20',
    '1_Geschwindigkeitsbeschraenkung_30',   
    '2_Geschwindigkeitsbeschraenkung_50',
    '3_Geschwindigkeitsbeschraenkung_60',  
    '4_Geschwindigkeitsbeschraenkung_70',
    '5_Geschwindigkeitsbeschraenkung_80',
    '6_Ende_Hoechstgeschwindigkeit',
    '7_Geschwindigkeitsbeschraenkung_100',
    '8_Geschwindigkeitsbeschraenkung_120',
    '9_Ueberholen_verboten',
    '10_Ueberholverbot_fuer_Kraftfahrzeuge',
    '11_Vorfahrt',
    '12_Hauptstrasse',
    '13_Vorfahrt_gewaehren',
    '14_Stop',
    '15_Farhverbot',
    '16_Fahrverbot_fuer_Kraftfahrzeuge',
    '17_Verbot_der_Einfahrt',
    '18_Gefahrstelle',
    '19_Kurve_links',
    '20_Kurve_rechts',
    '21_Doppelkurve_zunaechst_links',
    '22_Uneben_Fahrbahn',
    '23_Schleuder_oder_Rutschgefahr',
    '24_Verengung_rechts',
    '25_Baustelle',
    '26_Lichtzeichenanlage',
    '27_Fussgaenger',
    '28_Kinder',
    '29_Radverkehr',
    '30_Schnee_oder_Eisglaette',
    '31_Wildwechsel',
    '32_Ende_Geschwindigkeitsbegraenzungen',
    '33_Fahrtrichtung_rechts',
    '34_Fahrtrichtung_links',
    '35_Fahrtrichtung_geradeaus',
    '36_Fahrtrichtung_geradeaus_rechts',
    '37_Fahrtrichtung_geradeaus_links',
    '38_Vorbeifahrt_rechts',
    '39_Vorbeifahrt_links',
    '40_Kreisverkehr',
    '41_Ende_Ueberholverbot',
    '42_Ende_Ueberholverbot_fuer_Kraftfahrzeuge'
]

print(f"ASTRA - Train: {len(astra_train)}, Val: {len(astra_val)}, Test: {len(astra_test)}, Classes: {astra_num_classes}")
print(f"GTSRB - Train: {len(gtsrb_train)}, Val: {len(gtsrb_val)}, Test: {len(gtsrb_test)}, Classes: {gtsrb_num_classes}")

timer.end_phase()

# ==========================
# 7. CNN Model
# ==========================
timer.start_phase("Model Definition")

class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes):
        super(TrafficSignCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

timer.end_phase()

# ==========================
# 8. Train ASTRA Model
# ==========================
timer.start_phase("ASTRA Model Training")

print(f"\n{'='*50}")
print(f"Training ASTRA Model")
print(f"{'='*50}")

# ASTRA DataLoaders
batch_size = 32
astra_train_loader = DataLoader(astra_train, batch_size=batch_size, shuffle=True)
astra_val_loader   = DataLoader(astra_val, batch_size=batch_size)
astra_test_loader  = DataLoader(astra_test, batch_size=batch_size)

# ASTRA Model, Loss, Optimizer
astra_model = TrafficSignCNN(astra_num_classes).to(device)
astra_criterion = nn.CrossEntropyLoss()
astra_optimizer = optim.Adam(astra_model.parameters(), lr=0.001, weight_decay=1e-4)
astra_scheduler = optim.lr_scheduler.StepLR(astra_optimizer, step_size=5, gamma=0.7)

astra_best_val_acc = 0.0
epochs = 25

# ASTRA Training loop
for epoch in range(epochs):
    # Training
    astra_model.train()
    running_loss, correct = 0.0, 0
    for images, labels in astra_train_loader:
        images, labels = images.to(device), labels.to(device)

        astra_optimizer.zero_grad()
        outputs = astra_model(images)
        loss = astra_criterion(outputs, labels)
        loss.backward()
        astra_optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    train_loss = running_loss / len(astra_train)
    train_acc = correct / len(astra_train)

    # Validation
    astra_model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad(): # Gradient-Tracking is disabled during validation (PyTorch tracks gradients by default). This saves memory and speeds up the validation process.
        for images, labels in astra_val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = astra_model(images)
            loss = astra_criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss /= len(astra_val)
    val_acc = val_correct / len(astra_val)
    
    astra_scheduler.step()
    
    if val_acc > astra_best_val_acc:
        astra_best_val_acc = val_acc
        astra_best_model_state = astra_model.state_dict().copy()

    print(f"Epoch {epoch+1:2d}/{epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
          f"LR: {astra_optimizer.param_groups[0]['lr']:.6f}")

# Load best ASTRA model for testing
astra_model.load_state_dict(astra_best_model_state)

timer.end_phase()

# ==========================
# ASTRA Testing and Evaluation
# ==========================
timer.start_phase("ASTRA Model Evaluation")

# ASTRA Testing with detailed metrics
astra_model.eval()
astra_all_preds = []
astra_all_labels = []
astra_all_confidences = []

with torch.no_grad():
    for images, labels in astra_test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = astra_model(images)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
        
        astra_all_preds.extend(predictions.cpu().numpy())
        astra_all_labels.extend(labels.cpu().numpy())
        astra_all_confidences.extend(confidences.cpu().numpy())

astra_test_acc = sum(np.array(astra_all_preds) == np.array(astra_all_labels)) / len(astra_all_labels)

# Calculate ASTRA metrics
astra_precision, astra_recall, astra_f1, _ = precision_recall_fscore_support(
    astra_all_labels, astra_all_preds, average='weighted', zero_division=0
)

# Calculate class-wise confidence for ASTRA
astra_class_confidences = {}
for class_idx in range(astra_num_classes):
    class_mask = np.array(astra_all_labels) == class_idx
    if np.sum(class_mask) > 0:
        astra_class_confidences[class_idx] = np.mean(np.array(astra_all_confidences)[class_mask])
    else:
        astra_class_confidences[class_idx] = 0.0

print(f"\nASTRA Detailed Metrics:")
print(f"Best Validation Accuracy: {astra_best_val_acc:.4f}")
print(f"Test Accuracy: {astra_test_acc:.4f}")
print(f"Precision: {astra_precision:.4f}")
print(f"Recall: {astra_recall:.4f}")
print(f"F1-Score: {astra_f1:.4f}")
print(f"Average Confidence: {np.mean(astra_all_confidences):.4f}")

# ASTRA Confusion matrix
plt.figure(figsize=(12, 10))
astra_cm = confusion_matrix(astra_all_labels, astra_all_preds)
sns.heatmap(astra_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[name[:20] + '...' if len(name) > 20 else name for name in astra_class_names],
            yticklabels=[name[:20] + '...' if len(name) > 20 else name for name in astra_class_names])
plt.title('ASTRA Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(os.path.expanduser("~"), "Desktop", "astra_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()

# ASTRA Extended classification report with confidence
print(f"\nASTRA Classification Report with Confidence:")
astra_report = classification_report(astra_all_labels, astra_all_preds, 
                                   target_names=astra_class_names, 
                                   output_dict=True, zero_division=0)

print(f"{'Class':<40} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10} {'Confidence':<10}")
print("-" * 100)
for i, class_name in enumerate(astra_class_names):
    if class_name in astra_report:
        precision = astra_report[class_name]['precision']
        recall = astra_report[class_name]['recall']
        f1 = astra_report[class_name]['f1-score']
        support = astra_report[class_name]['support']
        confidence = astra_class_confidences[i]
        print(f"{class_name[:39]:<40} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10.0f} {confidence:<10.3f}")

print(f"\n{'Average/Total':<40} {astra_precision:<10.3f} {astra_recall:<10.3f} {astra_f1:<10.3f} {len(astra_all_labels):<10.0f} {np.mean(astra_all_confidences):<10.3f}")

# Save ASTRA model
astra_model_path = os.path.join(os.path.expanduser("~"), "Desktop", "traffic_sign_cnn_astra.pth")
torch.save(astra_best_model_state, astra_model_path)
print(f"ASTRA Model saved to {astra_model_path}")

timer.end_phase()

# ==========================
# 9. Train GTSRB Model
# ==========================
timer.start_phase("GTSRB Model Training")

print(f"\n{'='*50}")
print(f"Training GTSRB Model")
print(f"{'='*50}")

# GTSRB DataLoaders
gtsrb_train_loader = DataLoader(gtsrb_train, batch_size=batch_size, shuffle=True)
gtsrb_val_loader   = DataLoader(gtsrb_val, batch_size=batch_size)
gtsrb_test_loader  = DataLoader(gtsrb_test, batch_size=batch_size)

# GTSRB Model, Loss, Optimizer
gtsrb_model = TrafficSignCNN(gtsrb_num_classes).to(device)
gtsrb_criterion = nn.CrossEntropyLoss()
gtsrb_optimizer = optim.Adam(gtsrb_model.parameters(), lr=0.001, weight_decay=1e-4)
gtsrb_scheduler = optim.lr_scheduler.StepLR(gtsrb_optimizer, step_size=5, gamma=0.7)

gtsrb_best_val_acc = 0.0

# GTSRB Training loop
for epoch in range(epochs):
    # Training
    gtsrb_model.train()
    running_loss, correct = 0.0, 0
    for images, labels in gtsrb_train_loader:
        images, labels = images.to(device), labels.to(device)

        gtsrb_optimizer.zero_grad()
        outputs = gtsrb_model(images)
        loss = gtsrb_criterion(outputs, labels)
        loss.backward()
        gtsrb_optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    train_loss = running_loss / len(gtsrb_train)
    train_acc = correct / len(gtsrb_train)

    # Validation
    gtsrb_model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in gtsrb_val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = gtsrb_model(images)
            loss = gtsrb_criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss /= len(gtsrb_val)
    val_acc = val_correct / len(gtsrb_val)
    
    gtsrb_scheduler.step()
    
    if val_acc > gtsrb_best_val_acc:
        gtsrb_best_val_acc = val_acc
        gtsrb_best_model_state = gtsrb_model.state_dict().copy()

    print(f"Epoch {epoch+1:2d}/{epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
          f"LR: {gtsrb_optimizer.param_groups[0]['lr']:.6f}")

# Load best GTSRB model for testing
gtsrb_model.load_state_dict(gtsrb_best_model_state)

timer.end_phase()

# ==========================
# GTSRB Testing and Evaluation
# ==========================
timer.start_phase("GTSRB Model Evaluation")

# GTSRB Testing with detailed metrics
gtsrb_model.eval()
gtsrb_all_preds = []
gtsrb_all_labels = []
gtsrb_all_confidences = []

with torch.no_grad():
    for images, labels in gtsrb_test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = gtsrb_model(images)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
        
        gtsrb_all_preds.extend(predictions.cpu().numpy())
        gtsrb_all_labels.extend(labels.cpu().numpy())
        gtsrb_all_confidences.extend(confidences.cpu().numpy())

gtsrb_test_acc = sum(np.array(gtsrb_all_preds) == np.array(gtsrb_all_labels)) / len(gtsrb_all_labels)

# Calculate GTSRB metrics
gtsrb_precision, gtsrb_recall, gtsrb_f1, _ = precision_recall_fscore_support(
    gtsrb_all_labels, gtsrb_all_preds, average='weighted', zero_division=0
)

# Calculate class-wise confidence for GTSRB
gtsrb_class_confidences = {}
for class_idx in range(gtsrb_num_classes):
    class_mask = np.array(gtsrb_all_labels) == class_idx
    if np.sum(class_mask) > 0:
        gtsrb_class_confidences[class_idx] = np.mean(np.array(gtsrb_all_confidences)[class_mask])
    else:
        gtsrb_class_confidences[class_idx] = 0.0

print(f"\nGTSRB Detailed Metrics:")
print(f"Best Validation Accuracy: {gtsrb_best_val_acc:.4f}")
print(f"Test Accuracy: {gtsrb_test_acc:.4f}")
print(f"Precision: {gtsrb_precision:.4f}")
print(f"Recall: {gtsrb_recall:.4f}")
print(f"F1-Score: {gtsrb_f1:.4f}")
print(f"Average Confidence: {np.mean(gtsrb_all_confidences):.4f}")

# GTSRB Confusion matrix
plt.figure(figsize=(16, 14))
gtsrb_cm = confusion_matrix(gtsrb_all_labels, gtsrb_all_preds)
sns.heatmap(gtsrb_cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=[f"{i}: {name[:15]}..." if len(name) > 15 else f"{i}: {name}" 
                        for i, name in enumerate(gtsrb_class_names)],
            yticklabels=[f"{i}: {name[:15]}..." if len(name) > 15 else f"{i}: {name}" 
                        for i, name in enumerate(gtsrb_class_names)])
plt.title('GTSRB Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(os.path.expanduser("~"), "Desktop", "gtsrb_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()

# GTSRB Extended classification report with confidence
print(f"\nGTSRB Classification Report with Confidence:")
gtsrb_report = classification_report(gtsrb_all_labels, gtsrb_all_preds, 
                                   target_names=gtsrb_class_names, 
                                   output_dict=True, zero_division=0)

print(f"{'Class':<35} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10} {'Confidence':<10}")
print("-" * 95)
for i, class_name in enumerate(gtsrb_class_names):
    if class_name in gtsrb_report:
        precision = gtsrb_report[class_name]['precision']
        recall = gtsrb_report[class_name]['recall']
        f1 = gtsrb_report[class_name]['f1-score']
        support = gtsrb_report[class_name]['support']
        confidence = gtsrb_class_confidences[i]
        print(f"{class_name[:34]:<35} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10.0f} {confidence:<10.3f}")

print(f"\n{'Average/Total':<35} {gtsrb_precision:<10.3f} {gtsrb_recall:<10.3f} {gtsrb_f1:<10.3f} {len(gtsrb_all_labels):<10.0f} {np.mean(gtsrb_all_confidences):<10.3f}")

# Save GTSRB model
gtsrb_model_path = os.path.join(os.path.expanduser("~"), "Desktop", "traffic_sign_cnn_gtsrb.pth")
torch.save(gtsrb_best_model_state, gtsrb_model_path)
print(f"GTSRB Model saved to {gtsrb_model_path}")

timer.end_phase()

# ==========================
# 10. Final Summary
# ==========================
timer.start_phase("Final Analysis and Reports")

print(f"\n{'='*80}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*80}")
print(f"ASTRA Model:")
print(f"  Test Accuracy: {astra_test_acc:.4f}")
print(f"  Precision: {astra_precision:.4f}")
print(f"  Recall: {astra_recall:.4f}")
print(f"  F1-Score: {astra_f1:.4f}")
print(f"  Average Confidence: {np.mean(astra_all_confidences):.4f}")
print(f"\nGTSRB Model:")
print(f"  Test Accuracy: {gtsrb_test_acc:.4f}")
print(f"  Precision: {gtsrb_precision:.4f}")
print(f"  Recall: {gtsrb_recall:.4f}")
print(f"  F1-Score: {gtsrb_f1:.4f}")
print(f"  Average Confidence: {np.mean(gtsrb_all_confidences):.4f}")
print(f"{'='*80}")

# Confidence distribution analysis
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist(astra_all_confidences, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('ASTRA Confidence Distribution')
plt.xlabel('Confidence')
plt.ylabel('Number of Predictions')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(gtsrb_all_confidences, bins=50, alpha=0.7, color='orange', edgecolor='black')
plt.title('GTSRB Confidence Distribution')
plt.xlabel('Confidence')
plt.ylabel('Number of Predictions')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.expanduser("~"), "Desktop", "confidence_distributions.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"\nConfusion matrices and confidence distributions saved to desktop:")
print(f"- astra_confusion_matrix.png")
print(f"- gtsrb_confusion_matrix.png") 
print(f"- confidence_distributions.png")

# ==========================
# 11. Performance Benchmark for Real-Life Deployment
# ==========================
print(f"\n{'='*80}")
print("PERFORMANCE BENCHMARK FOR REAL-LIFE DEPLOYMENT")
print(f"{'='*80}")

# Benchmark settings
num_warmup = 50
num_test = 1000
batch_sizes = [1, 8, 16, 32]

def benchmark_model(model, model_name, test_loader, device):
    print(f"\nBenchmark for {model_name} Model:")
    print(f"Device: {device}")
    model.eval()
    
    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        
        dummy_input = torch.randn(batch_size, 3, 64, 64).to(device)
        
        # Warm-up: Fisrt few CUDA runs are often slower due to initialization (Kernel-Loading) and PyTorch has to allocate memory for the model.
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Latency measurement
        latencies = []
        with torch.no_grad():
            for _ in range(num_test):
                start_time = time.perf_counter()
                outputs = model(dummy_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                latencies.append(end_time - start_time)
        
        # Calculate statistics
        avg_latency = np.mean(latencies) * 1000  # in ms
        min_latency = np.min(latencies) * 1000
        max_latency = np.max(latencies) * 1000
        p95_latency = np.percentile(latencies, 95) * 1000 # 95% of all requests are faster than this value; 5% are slower than this value
        p99_latency = np.percentile(latencies, 99) * 1000 # 99% of all requests are faster than this value; 1% (Worst Case) are slower than this value --> Example: P99 = 45ms --> 1% have a latency of more than 45ms (problematic for real-time applications)
        
        # Calculate throughput
        images_per_second = batch_size / (avg_latency / 1000)
        
        print(f"  Latency (ms):")
        print(f"    - Average: {avg_latency:.2f} ms")
        print(f"    - Minimum: {min_latency:.2f} ms")
        print(f"    - Maximum: {max_latency:.2f} ms")
        print(f"    - P95: {p95_latency:.2f} ms")
        print(f"    - P99: {p99_latency:.2f} ms")
        print(f"  Throughput: {images_per_second:.1f} images/second")

# Benchmark both models
benchmark_model(astra_model, "ASTRA", astra_test_loader, device)
benchmark_model(gtsrb_model, "GTSRB", gtsrb_test_loader, device)

# Model size analysis
print(f"\n{'='*60}")
print("MODEL SIZE ANALYSIS")
print(f"{'='*60}")

def analyze_model_size(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_size = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    
    print(f"\n{model_name} Model Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Estimated size: {param_size:.2f} MB")

analyze_model_size(astra_model, "ASTRA")
analyze_model_size(gtsrb_model, "GTSRB")

timer.end_phase()

# ==========================
# MONITORING COMPLETION AND REPORT GENERATION
# ==========================
timer.start_phase("Monitoring Reports and Cleanup")

# Stop monitoring
monitor.stop_monitoring()
timer.stop()

# Generate reports
print(timer.get_summary())
print(monitor.get_summary())

# Create and save plots
monitor.plot_resources(save_path=os.path.join(os.path.expanduser("~"), "Desktop", "cnn_training_resources.png"))

# Save monitoring data as JSON
monitoring_data = {
    'script_name': 'CNN Training for ASTRA and GTSRB',
    'total_runtime': float(timer.end_time - timer.start_time),
    'phase_times': {k: float(v) for k, v in timer.phase_times.items()},
    'resource_data': {
        'timestamps': [float(x) for x in monitor.data['timestamps']],
        'cpu_percent': [float(x) for x in monitor.data['cpu_percent']],
        'ram_percent': [float(x) for x in monitor.data['ram_percent']],
        'ram_used_gb': [float(x) for x in monitor.data['ram_used_gb']],
        'gpu_percent': [float(x) for x in monitor.data['gpu_percent']],
        'gpu_memory_percent': [float(x) for x in monitor.data['gpu_memory_percent']],
        'gpu_memory_used_gb': [float(x) for x in monitor.data['gpu_memory_used_gb']]
    },
    'results': {
        'astra': {
            'test_accuracy': float(astra_test_acc),
            'precision': float(astra_precision),
            'recall': float(astra_recall),
            'f1_score': float(astra_f1),
            'avg_confidence': float(np.mean(astra_all_confidences))
        },
        'gtsrb': {
            'test_accuracy': float(gtsrb_test_acc),
            'precision': float(gtsrb_precision),
            'recall': float(gtsrb_recall),
            'f1_score': float(gtsrb_f1),
            'avg_confidence': float(np.mean(gtsrb_all_confidences))
        }
    },
    'summary': {
        'avg_cpu': float(np.mean(monitor.data['cpu_percent'])),
        'max_cpu': float(np.max(monitor.data['cpu_percent'])),
        'avg_ram_gb': float(np.mean(monitor.data['ram_used_gb'])),
        'max_ram_gb': float(np.max(monitor.data['ram_used_gb'])),
        'avg_gpu': float(np.mean(monitor.data['gpu_percent'])),
        'max_gpu': float(np.max(monitor.data['gpu_percent'])),
        'avg_vram_gb': float(np.mean(monitor.data['gpu_memory_used_gb'])),
        'max_vram_gb': float(np.max(monitor.data['gpu_memory_used_gb']))
    }
}

# Save JSON report
json_report_path = os.path.join(os.path.expanduser("~"), "Desktop", "cnn_training_monitoring_report.json")
with open(json_report_path, 'w', encoding='utf-8') as f:
    json.dump(monitoring_data, f, indent=2, ensure_ascii=False)

print(f"\nMonitoring report saved: {json_report_path}")
print(f"Resource plot saved: {os.path.join(os.path.expanduser('~'), 'Desktop', 'cnn_training_resources.png')}")

print(f"\n{'='*80}")

timer.end_phase()

print(f"\nCNN TRAINING WITH MONITORING SUCCESSFULLY COMPLETED")
print(f"All models, reports and logs have been saved to desktop.")
print(f"Saved files:")
print(f"   - traffic_sign_cnn_astra.pth")
print(f"   - traffic_sign_cnn_gtsrb.pth")
print(f"   - astra_confusion_matrix.png")
print(f"   - gtsrb_confusion_matrix.png")
print(f"   - confidence_distributions.png")
print(f"   - cnn_training_resources.png")
print(f"   - cnn_training_monitoring_report.json")
