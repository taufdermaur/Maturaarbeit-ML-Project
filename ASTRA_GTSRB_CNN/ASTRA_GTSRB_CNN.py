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

# ==========================
# 1. Reproducibility
# ==========================
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
    transforms.RandomRotation(15),  # Rotation um ±15 Grad
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==========================
# 4. Custom Dataset für GTSRB Testing
# ==========================
class GTSRBTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, train_classes=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.train_classes = train_classes
        
        # Erstelle Mapping von CSV ClassId zu Training Label
        if train_classes is not None:
            self.label_mapping = {}
            # Die Trainingsordner sind nach ClassId benannt (0_, 1_, 2_, etc.)
            for folder_name in train_classes:
                class_id = int(folder_name.split('_')[0])  # Extrahiere Zahl vom Ordnernamen
                train_label = train_classes.index(folder_name)  # Index im sortierten Training
                self.label_mapping[class_id] = train_label
        else:
            self.label_mapping = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['Path'])
        image = Image.open(img_path).convert('RGB')
        csv_class_id = self.data.iloc[idx]['ClassId']
        
        # Konvertiere CSV ClassId zum Training Label
        if self.label_mapping is not None and csv_class_id in self.label_mapping:
            label = self.label_mapping[csv_class_id]
        else:
            label = csv_class_id  # Fallback
        
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

# Anzahl Klassen für jeden Datensatz
astra_num_classes = len(astra_train_full.classes)
gtsrb_num_classes = len(gtsrb_train_full.classes)

# Deutsche Klassennamen für ASTRA
astra_class_names = [
    "Besondere Signale",
    "Ergänzende Angaben zu Signalen", 
    "Fahranordnungen Parkierungsbeschränkungen",
    "Fahrverbote Mass und Gewichtsbeschränkungen",
    "Informationshinweise",
    "Markierungen und Leiteinrichtungen",
    "Verhaltenshinweise", 
    "Vortrittssignale",
    "Wegweisung auf Autobahnen und Autostrassen",
    "Wegweisung auf Haupt und Nebenstrassen"
]

# Deutsche Klassennamen für GTSRB
gtsrb_class_names = [
    "Geschwindigkeitsbeschränkung 20",
    "Geschwindigkeitsbeschränkung 30", 
    "Geschwindigkeitsbeschränkung 50",
    "Geschwindigkeitsbeschränkung 60",
    "Geschwindigkeitsbeschränkung 70",
    "Geschwindigkeitsbeschränkung 80",
    "Ende Höchstgeschwindigkeit",
    "Geschwindigkeitsbeschränkung 100",
    "Geschwindigkeitsbeschränkung 120",
    "Überholen verboten",
    "Überholverbot für Kraftfahrzeuge",
    "Vorfahrt",
    "Hauptstraße",
    "Vorfahrt gewähren",
    "Stop",
    "Fahrverbot",
    "Fahrverbot für Kraftfahrzeuge",
    "Verbot der Einfahrt",
    "Gefahrstelle",
    "Kurve links",
    "Kurve rechts",
    "Doppelkurve zunächst links",
    "Unebene Fahrbahn",
    "Schleuder oder Rutschgefahr",
    "Verengung rechts",
    "Baustelle",
    "Lichtzeichenanlage",
    "Fußgänger",
    "Kinder",
    "Radverkehr",
    "Schnee oder Eisglätte",
    "Wildwechsel",
    "Ende Geschwindigkeitsbegrenzungen",
    "Fahrtrichtung rechts",
    "Fahrtrichtung links",
    "Fahrtrichtung geradeaus",
    "Fahrtrichtung geradeaus rechts",
    "Fahrtrichtung geradeaus links",
    "Vorbeifahrt rechts",
    "Vorbeifahrt links",
    "Kreisverkehr",
    "Ende Überholverbot",
    "Ende Überholverbot für Kraftfahrzeuge"
]

print(f"ASTRA - Train: {len(astra_train)}, Val: {len(astra_val)}, Test: {len(astra_test)}, Classes: {astra_num_classes}")
print(f"GTSRB - Train: {len(gtsrb_train)}, Val: {len(gtsrb_val)}, Test: {len(gtsrb_test)}, Classes: {gtsrb_num_classes}")

# ==========================
# 7. CNN Model
# ==========================
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes):
        super(TrafficSignCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # → 32x64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 32x32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 64x16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # → 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 128x8x8
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # → 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  # → 256x4x4
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

# ==========================
# 8. Train ASTRA Model
# ==========================
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
    with torch.no_grad():
        for images, labels in astra_val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = astra_model(images)
            loss = astra_criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss /= len(astra_val)
    val_acc = val_correct / len(astra_val)
    
    # Learning rate scheduler
    astra_scheduler.step()
    
    # Save best model
    if val_acc > astra_best_val_acc:
        astra_best_val_acc = val_acc
        astra_best_model_state = astra_model.state_dict().copy()

    print(f"Epoch {epoch+1:2d}/{epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
          f"LR: {astra_optimizer.param_groups[0]['lr']:.6f}")

# Load best ASTRA model for testing
astra_model.load_state_dict(astra_best_model_state)

# ASTRA Testing mit detaillierten Metriken
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

# ASTRA Metriken berechnen
astra_precision, astra_recall, astra_f1, _ = precision_recall_fscore_support(
    astra_all_labels, astra_all_preds, average='weighted', zero_division=0
)

# ASTRA klassenweise Konfidenz berechnen
astra_class_confidences = {}
for class_idx in range(astra_num_classes):
    class_mask = np.array(astra_all_labels) == class_idx
    if np.sum(class_mask) > 0:
        astra_class_confidences[class_idx] = np.mean(np.array(astra_all_confidences)[class_mask])
    else:
        astra_class_confidences[class_idx] = 0.0

print(f"\nASTRA Detaillierte Metriken:")
print(f"Best Validation Accuracy: {astra_best_val_acc:.4f}")
print(f"Test Accuracy: {astra_test_acc:.4f}")
print(f"Präzision (Precision): {astra_precision:.4f}")
print(f"Sensitivität (Recall): {astra_recall:.4f}")
print(f"F1-Score: {astra_f1:.4f}")
print(f"Durchschnittliche Konfidenz: {np.mean(astra_all_confidences):.4f}")

# ASTRA Konfusionsmatrix
plt.figure(figsize=(12, 10))
astra_cm = confusion_matrix(astra_all_labels, astra_all_preds)
sns.heatmap(astra_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[name[:20] + '...' if len(name) > 20 else name for name in astra_class_names],
            yticklabels=[name[:20] + '...' if len(name) > 20 else name for name in astra_class_names])
plt.title('ASTRA Konfusionsmatrix')
plt.xlabel('Vorhergesagte Klasse')
plt.ylabel('Wahre Klasse')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(os.path.expanduser("~"), "Desktop", "astra_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()

# ASTRA Erweiteter Klassifikationsbericht mit Konfidenz
print(f"\nASTRA Klassifikationsbericht mit Konfidenz:")
astra_report = classification_report(astra_all_labels, astra_all_preds, 
                                   target_names=astra_class_names, 
                                   output_dict=True, zero_division=0)

print(f"{'Klasse':<40} {'Präzision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10} {'Konfidenz':<10}")
print("-" * 100)
for i, class_name in enumerate(astra_class_names):
    if class_name in astra_report:
        precision = astra_report[class_name]['precision']
        recall = astra_report[class_name]['recall']
        f1 = astra_report[class_name]['f1-score']
        support = astra_report[class_name]['support']
        confidence = astra_class_confidences[i]
        print(f"{class_name[:39]:<40} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10.0f} {confidence:<10.3f}")

print(f"\n{'Durchschnitt/Gesamt':<40} {astra_precision:<10.3f} {astra_recall:<10.3f} {astra_f1:<10.3f} {len(astra_all_labels):<10.0f} {np.mean(astra_all_confidences):<10.3f}")

# Save ASTRA model
astra_model_path = os.path.join(os.path.expanduser("~"), "Desktop", "traffic_sign_cnn_astra.pth")
torch.save(astra_best_model_state, astra_model_path)
print(f"ASTRA Model saved to {astra_model_path}")

# ==========================
# 9. Train GTSRB Model
# ==========================
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
    
    # Learning rate scheduler
    gtsrb_scheduler.step()
    
    # Save best model
    if val_acc > gtsrb_best_val_acc:
        gtsrb_best_val_acc = val_acc
        gtsrb_best_model_state = gtsrb_model.state_dict().copy()

    print(f"Epoch {epoch+1:2d}/{epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
          f"LR: {gtsrb_optimizer.param_groups[0]['lr']:.6f}")

# Load best GTSRB model for testing
gtsrb_model.load_state_dict(gtsrb_best_model_state)

# GTSRB Testing mit detaillierten Metriken
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

# GTSRB Metriken berechnen
gtsrb_precision, gtsrb_recall, gtsrb_f1, _ = precision_recall_fscore_support(
    gtsrb_all_labels, gtsrb_all_preds, average='weighted', zero_division=0
)

# GTSRB klassenweise Konfidenz berechnen
gtsrb_class_confidences = {}
for class_idx in range(gtsrb_num_classes):
    class_mask = np.array(gtsrb_all_labels) == class_idx
    if np.sum(class_mask) > 0:
        gtsrb_class_confidences[class_idx] = np.mean(np.array(gtsrb_all_confidences)[class_mask])
    else:
        gtsrb_class_confidences[class_idx] = 0.0

print(f"\nGTSRB Detaillierte Metriken:")
print(f"Best Validation Accuracy: {gtsrb_best_val_acc:.4f}")
print(f"Test Accuracy: {gtsrb_test_acc:.4f}")
print(f"Präzision (Precision): {gtsrb_precision:.4f}")
print(f"Sensitivität (Recall): {gtsrb_recall:.4f}")
print(f"F1-Score: {gtsrb_f1:.4f}")
print(f"Durchschnittliche Konfidenz: {np.mean(gtsrb_all_confidences):.4f}")

# GTSRB Konfusionsmatrix
plt.figure(figsize=(16, 14))
gtsrb_cm = confusion_matrix(gtsrb_all_labels, gtsrb_all_preds)
sns.heatmap(gtsrb_cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=[f"{i}: {name[:15]}..." if len(name) > 15 else f"{i}: {name}" 
                        for i, name in enumerate(gtsrb_class_names)],
            yticklabels=[f"{i}: {name[:15]}..." if len(name) > 15 else f"{i}: {name}" 
                        for i, name in enumerate(gtsrb_class_names)])
plt.title('GTSRB Konfusionsmatrix')
plt.xlabel('Vorhergesagte Klasse')
plt.ylabel('Wahre Klasse')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(os.path.expanduser("~"), "Desktop", "gtsrb_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()

# GTSRB Erweiteter Klassifikationsbericht mit Konfidenz
print(f"\nGTSRB Klassifikationsbericht mit Konfidenz:")
gtsrb_report = classification_report(gtsrb_all_labels, gtsrb_all_preds, 
                                   target_names=gtsrb_class_names, 
                                   output_dict=True, zero_division=0)

print(f"{'Klasse':<35} {'Präzision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10} {'Konfidenz':<10}")
print("-" * 95)
for i, class_name in enumerate(gtsrb_class_names):
    if class_name in gtsrb_report:
        precision = gtsrb_report[class_name]['precision']
        recall = gtsrb_report[class_name]['recall']
        f1 = gtsrb_report[class_name]['f1-score']
        support = gtsrb_report[class_name]['support']
        confidence = gtsrb_class_confidences[i]
        print(f"{class_name[:34]:<35} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10.0f} {confidence:<10.3f}")

print(f"\n{'Durchschnitt/Gesamt':<35} {gtsrb_precision:<10.3f} {gtsrb_recall:<10.3f} {gtsrb_f1:<10.3f} {len(gtsrb_all_labels):<10.0f} {np.mean(gtsrb_all_confidences):<10.3f}")

# Save GTSRB model
gtsrb_model_path = os.path.join(os.path.expanduser("~"), "Desktop", "traffic_sign_cnn_gtsrb.pth")
torch.save(gtsrb_best_model_state, gtsrb_model_path)
print(f"GTSRB Model saved to {gtsrb_model_path}")

# ==========================
# 10. Final Summary
# ==========================
print(f"\n{'='*80}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*80}")
print(f"ASTRA Model:")
print(f"  Test Accuracy: {astra_test_acc:.4f}")
print(f"  Präzision: {astra_precision:.4f}")
print(f"  Sensitivität: {astra_recall:.4f}")
print(f"  F1-Score: {astra_f1:.4f}")
print(f"  Durchschnittliche Konfidenz: {np.mean(astra_all_confidences):.4f}")
print(f"\nGTSRB Model:")
print(f"  Test Accuracy: {gtsrb_test_acc:.4f}")
print(f"  Präzision: {gtsrb_precision:.4f}")
print(f"  Sensitivität: {gtsrb_recall:.4f}")
print(f"  F1-Score: {gtsrb_f1:.4f}")
print(f"  Durchschnittliche Konfidenz: {np.mean(gtsrb_all_confidences):.4f}")
print(f"{'='*80}")

# Konfidenz-Verteilungsanalyse
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist(astra_all_confidences, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('ASTRA Konfidenz-Verteilung')
plt.xlabel('Konfidenz')
plt.ylabel('Anzahl Vorhersagen')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(gtsrb_all_confidences, bins=50, alpha=0.7, color='orange', edgecolor='black')
plt.title('GTSRB Konfidenz-Verteilung')
plt.xlabel('Konfidenz')
plt.ylabel('Anzahl Vorhersagen')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.expanduser("~"), "Desktop", "confidence_distributions.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"\nKonfusionsmatrizen und Konfidenz-Verteilungen wurden auf dem Desktop gespeichert:")
print(f"- astra_confusion_matrix.png")
print(f"- gtsrb_confusion_matrix.png") 
print(f"- confidence_distributions.png")

# ==========================
# 11. Performance Benchmark für Real-Life Deployment
# ==========================
import time

print(f"\n{'='*80}")
print("PERFORMANCE BENCHMARK FÜR REAL-LIFE DEPLOYMENT")
print(f"{'='*80}")

# Benchmark-Einstellungen
num_warmup = 50    # Warm-up Durchläufe
num_test = 1000    # Test-Durchläufe für Messung
batch_sizes = [1, 8, 16, 32]  # Verschiedene Batch-Größen

def benchmark_model(model, model_name, test_loader, device):
    print(f"\nBenchmark für {model_name} Modell:")
    print(f"Device: {device}")
    model.eval()
    
    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        
        # Erstelle Test-Batch
        dummy_input = torch.randn(batch_size, 3, 64, 64).to(device)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)
        
        # Synchronize GPU if using CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Latenz-Messung (Einzelne Inferenz)
        latencies = []
        with torch.no_grad():
            for _ in range(num_test):
                start_time = time.perf_counter()
                outputs = model(dummy_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                latencies.append(end_time - start_time)
        
        # Statistiken berechnen
        avg_latency = np.mean(latencies) * 1000  # in ms
        min_latency = np.min(latencies) * 1000
        max_latency = np.max(latencies) * 1000
        p95_latency = np.percentile(latencies, 95) * 1000
        p99_latency = np.percentile(latencies, 99) * 1000
        
        # Durchsatz berechnen
        images_per_second = batch_size / (avg_latency / 1000)
        
        print(f"  Latenz (ms):")
        print(f"    - Durchschnitt: {avg_latency:.2f} ms")
        print(f"    - Minimum: {min_latency:.2f} ms")
        print(f"    - Maximum: {max_latency:.2f} ms")
        print(f"    - P95: {p95_latency:.2f} ms")
        print(f"    - P99: {p99_latency:.2f} ms")
        print(f"  Durchsatz: {images_per_second:.1f} Bilder/Sekunde")
        
        # Real-Time Kategorisierung
        if avg_latency < 10:
            category = "Echzeit-tauglich (< 10ms)"
        elif avg_latency < 50:
            category = "Interaktiv (< 50ms)"
        elif avg_latency < 200:
            category = "Akzeptabel (< 200ms)"
        else:
            category = "Zu langsam (> 200ms)"
        
        print(f"  Bewertung: {category}")

# Benchmark beide Modelle
benchmark_model(astra_model, "ASTRA", astra_test_loader, device)
benchmark_model(gtsrb_model, "GTSRB", gtsrb_test_loader, device)

# Modell-Größen-Analyse
print(f"\n{'='*60}")
print("MODELL-GRÖSSEN ANALYSE")
print(f"{'='*60}")

def analyze_model_size(model, model_name):
    # Parameter zählen
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Modell-Größe schätzen (in MB)
    param_size = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    
    print(f"\n{model_name} Modell-Statistiken:")
    print(f"  - Gesamt Parameter: {total_params:,}")
    print(f"  - Trainierbare Parameter: {trainable_params:,}")
    print(f"  - Geschätzte Größe: {param_size:.2f} MB")

analyze_model_size(astra_model, "ASTRA")
analyze_model_size(gtsrb_model, "GTSRB")

print(f"\n{'='*80}")