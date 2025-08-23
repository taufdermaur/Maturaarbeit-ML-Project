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
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# ==========================
# ÜBERWACHUNGS-IMPORTS
# ==========================
import psutil
import threading
import json
from datetime import datetime
import time
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
                'astra': {'epochen': [], 'train_verlust': [], 'train_genauigkeit': [], 
                         'val_verlust': [], 'val_genauigkeit': [], 'epochen_zeit': []},
                'gtsrb': {'epochen': [], 'train_verlust': [], 'train_genauigkeit': [], 
                         'val_verlust': [], 'val_genauigkeit': [], 'epochen_zeit': []}
            },
            'test_metriken': {
                'astra': {}, 
                'gtsrb': {}
            },
            'konfusionsmatrizen': {
                'astra': [], 
                'gtsrb': []
            },
            'klassifikationsberichte': {
                'astra': {}, 
                'gtsrb': {}
            },
            'leistungsmetriken': {
                'astra': {}, 
                'gtsrb': {}
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
            
            # Training Metriken ASTRA
            if self.alle_daten['training_metriken']['astra']['epochen']:
                astra_training_df = pd.DataFrame(self.alle_daten['training_metriken']['astra'])
                astra_training_df.to_excel(writer, sheet_name='ASTRA_Training', index=False)
            
            # Training Metriken GTSRB
            if self.alle_daten['training_metriken']['gtsrb']['epochen']:
                gtsrb_training_df = pd.DataFrame(self.alle_daten['training_metriken']['gtsrb'])
                gtsrb_training_df.to_excel(writer, sheet_name='GTSRB_Training', index=False)
            
            # Test Metriken
            if self.alle_daten['test_metriken']['astra']:
                astra_test_df = pd.DataFrame([self.alle_daten['test_metriken']['astra']])
                astra_test_df.to_excel(writer, sheet_name='ASTRA_Test_Metriken', index=False)
            
            if self.alle_daten['test_metriken']['gtsrb']:
                gtsrb_test_df = pd.DataFrame([self.alle_daten['test_metriken']['gtsrb']])
                gtsrb_test_df.to_excel(writer, sheet_name='GTSRB_Test_Metriken', index=False)
            
            # Konfusionsmatrizen
            if self.alle_daten['konfusionsmatrizen']['astra']:
                astra_km_df = pd.DataFrame(self.alle_daten['konfusionsmatrizen']['astra'])
                astra_km_df.to_excel(writer, sheet_name='ASTRA_Konfusionsmatrix', index=False)
            
            if self.alle_daten['konfusionsmatrizen']['gtsrb']:
                gtsrb_km_df = pd.DataFrame(self.alle_daten['konfusionsmatrizen']['gtsrb'])
                gtsrb_km_df.to_excel(writer, sheet_name='GTSRB_Konfusionsmatrix', index=False)
            
            # Klassifikationsberichte
            if self.alle_daten['klassifikationsberichte']['astra']:
                astra_bericht_df = pd.DataFrame(self.alle_daten['klassifikationsberichte']['astra']).T
                astra_bericht_df.to_excel(writer, sheet_name='ASTRA_Klassifikation', index=True)
            
            if self.alle_daten['klassifikationsberichte']['gtsrb']:
                gtsrb_bericht_df = pd.DataFrame(self.alle_daten['klassifikationsberichte']['gtsrb']).T
                gtsrb_bericht_df.to_excel(writer, sheet_name='GTSRB_Klassifikation', index=True)
            
            # Leistungsmetriken
            if self.alle_daten['leistungsmetriken']['astra']:
                astra_leistung_df = pd.DataFrame([self.alle_daten['leistungsmetriken']['astra']])
                astra_leistung_df.to_excel(writer, sheet_name='ASTRA_Leistung', index=False)
            
            if self.alle_daten['leistungsmetriken']['gtsrb']:
                gtsrb_leistung_df = pd.DataFrame([self.alle_daten['leistungsmetriken']['gtsrb']])
                gtsrb_leistung_df.to_excel(writer, sheet_name='GTSRB_Leistung', index=False)
            
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
        """Überwachungsschleife (läuft in separatem Thread)"""
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
                
                time.sleep(2)  # Überwachung alle 2 Sekunden für Training
                
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
# HAUPTSKRIPT
# ==========================
print("CNN TRAINING SKRIPT - DATENSAMMLUNG")
print("="*60)

# Datensammler und Überwachung initialisieren
daten_sammler = DatenSammler()
zeitmesser = SkriptZeitmesser("CNN Training für ASTRA und GTSRB", daten_sammler)
monitor = RessourcenUeberwachung(daten_sammler)

# Überwachung starten
zeitmesser.starten()
monitor.ueberwachung_starten()

# ==========================
# 1. Setup und Gerätekonfiguration
# ==========================
zeitmesser.phase_starten("Setup und Initialisierung")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwendetes Gerät: {device}")

# Allgemeine Informationen sammeln
daten_sammler.alle_daten['allgemeine_info'] = {
    'datum': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'geraet': str(device),
    'seed': seed,
    'torch_version': torch.__version__,
    'cuda_verfuegbar': torch.cuda.is_available()
}

# Transformationen
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

zeitmesser.phase_beenden()

# ==========================
# 2. Datensatz-Setup
# ==========================
zeitmesser.phase_starten("Datensatzvorbereitung")

class GTSRBTestDataset(Dataset):
    def __init__(self, csv_datei, stammverzeichnis, transform=None, trainings_klassen=None):
        self.daten = pd.read_csv(csv_datei)
        self.stammverzeichnis = stammverzeichnis
        self.transform = transform
        self.trainings_klassen = trainings_klassen
        
        if trainings_klassen is not None:
            self.label_mapping = {}
            for ordner_name in trainings_klassen:
                klassen_id = int(ordner_name.split('_')[0])
                trainings_label = trainings_klassen.index(ordner_name)
                self.label_mapping[klassen_id] = trainings_label
        else:
            self.label_mapping = None
    
    def __len__(self):
        return len(self.daten)
    
    def __getitem__(self, idx):
        img_pfad = os.path.join(self.stammverzeichnis, self.daten.iloc[idx]['Path'])
        bild = Image.open(img_pfad).convert('RGB')
        csv_klassen_id = self.daten.iloc[idx]['ClassId']
        
        if self.label_mapping is not None and csv_klassen_id in self.label_mapping:
            label = self.label_mapping[csv_klassen_id]
        else:
            label = csv_klassen_id
        
        if self.transform:
            bild = self.transform(bild)
            
        return bild, label

# Datenpfade
astra_train_pfad = r"C:\Users\timau\Desktop\Datensätze\ASTRA\Train"
astra_test_pfad  = r"C:\Users\timau\Desktop\Datensätze\ASTRA\Test"
gtsrb_train_pfad = r"C:\Users\timau\Desktop\Datensätze\GTSRB\Train"
gtsrb_test_csv   = r"C:\Users\timau\Desktop\Datensätze\GTSRB\Test.csv"
gtsrb_basis_pfad  = r"C:\Users\timau\Desktop\Datensätze\GTSRB"

# Datensätze laden
astra_train_vollstaendig = datasets.ImageFolder(root=astra_train_pfad, transform=transform)
astra_test = datasets.ImageFolder(root=astra_test_pfad, transform=transform)
gtsrb_train_vollstaendig = datasets.ImageFolder(root=gtsrb_train_pfad, transform=transform)
gtsrb_test = GTSRBTestDataset(csv_datei=gtsrb_test_csv, stammverzeichnis=gtsrb_basis_pfad, 
                              transform=transform, trainings_klassen=gtsrb_train_vollstaendig.classes)

# Aufteilung in Train/Val
val_verhaeltnis = 0.2
astra_val_groesse = int(len(astra_train_vollstaendig) * val_verhaeltnis)
astra_train_groesse = len(astra_train_vollstaendig) - astra_val_groesse
astra_train, astra_val = random_split(astra_train_vollstaendig, [astra_train_groesse, astra_val_groesse], 
                                      generator=torch.Generator().manual_seed(seed))

gtsrb_val_groesse = int(len(gtsrb_train_vollstaendig) * val_verhaeltnis)
gtsrb_train_groesse = len(gtsrb_train_vollstaendig) - gtsrb_val_groesse
gtsrb_train, gtsrb_val = random_split(gtsrb_train_vollstaendig, [gtsrb_train_groesse, gtsrb_val_groesse], 
                                      generator=torch.Generator().manual_seed(seed))

astra_anzahl_klassen = len(astra_train_vollstaendig.classes)
gtsrb_anzahl_klassen = len(gtsrb_train_vollstaendig.classes)

# Datensatzgrößen in allgemeine Info
daten_sammler.alle_daten['allgemeine_info'].update({
    'astra_train_groesse': len(astra_train),
    'astra_val_groesse': len(astra_val),
    'astra_test_groesse': len(astra_test),
    'astra_anzahl_klassen': astra_anzahl_klassen,
    'gtsrb_train_groesse': len(gtsrb_train),
    'gtsrb_val_groesse': len(gtsrb_val),
    'gtsrb_test_groesse': len(gtsrb_test),
    'gtsrb_anzahl_klassen': gtsrb_anzahl_klassen
})

print(f"Datensätze geladen - ASTRA: {astra_anzahl_klassen} Klassen, GTSRB: {gtsrb_anzahl_klassen} Klassen")

zeitmesser.phase_beenden()

# ==========================
# 3. CNN-Modell Definition
# ==========================
zeitmesser.phase_starten("Modelldefinition")

class VerkehrszeichenCNN(nn.Module):
    def __init__(self, anzahl_klassen):
        super(VerkehrszeichenCNN, self).__init__()
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
            nn.Linear(256, anzahl_klassen)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

zeitmesser.phase_beenden()

# ==========================
# 4. ASTRA-Modell Training
# ==========================
zeitmesser.phase_starten("ASTRA-Modell Training")

print("ASTRA-Modell Training gestartet...")

batch_groesse = 32
astra_train_loader = DataLoader(astra_train, batch_size=batch_groesse, shuffle=True)
astra_val_loader = DataLoader(astra_val, batch_size=batch_groesse)
astra_test_loader = DataLoader(astra_test, batch_size=batch_groesse)

astra_modell = VerkehrszeichenCNN(astra_anzahl_klassen).to(device)
astra_verlustfunktion = nn.CrossEntropyLoss()
astra_optimierer = optim.Adam(astra_modell.parameters(), lr=0.001, weight_decay=1e-4)
astra_scheduler = optim.lr_scheduler.StepLR(astra_optimierer, step_size=5, gamma=0.7)

astra_beste_val_genauigkeit = 0.0
epochen = 25

# Training
for epoche in range(epochen):
    epoche_start_zeit = time.time()
    
    # Training
    astra_modell.train()
    laufender_verlust, korrekt = 0.0, 0
    for bilder, labels in astra_train_loader:
        bilder, labels = bilder.to(device), labels.to(device)
        astra_optimierer.zero_grad()
        ausgaben = astra_modell(bilder)
        verlust = astra_verlustfunktion(ausgaben, labels)
        verlust.backward()
        astra_optimierer.step()
        laufender_verlust += verlust.item() * bilder.size(0)
        korrekt += (ausgaben.argmax(1) == labels).sum().item()

    train_verlust = laufender_verlust / len(astra_train)
    train_genauigkeit = korrekt / len(astra_train)

    # Validierung
    astra_modell.eval()
    val_verlust, val_korrekt = 0.0, 0
    with torch.no_grad():
        for bilder, labels in astra_val_loader:
            bilder, labels = bilder.to(device), labels.to(device)
            ausgaben = astra_modell(bilder)
            verlust = astra_verlustfunktion(ausgaben, labels)
            val_verlust += verlust.item() * bilder.size(0)
            val_korrekt += (ausgaben.argmax(1) == labels).sum().item()

    val_verlust /= len(astra_val)
    val_genauigkeit = val_korrekt / len(astra_val)
    epoche_zeit = time.time() - epoche_start_zeit
    
    # Daten sammeln
    daten_sammler.alle_daten['training_metriken']['astra']['epochen'].append(epoche + 1)
    daten_sammler.alle_daten['training_metriken']['astra']['train_verlust'].append(train_verlust)
    daten_sammler.alle_daten['training_metriken']['astra']['train_genauigkeit'].append(train_genauigkeit)
    daten_sammler.alle_daten['training_metriken']['astra']['val_verlust'].append(val_verlust)
    daten_sammler.alle_daten['training_metriken']['astra']['val_genauigkeit'].append(val_genauigkeit)
    daten_sammler.alle_daten['training_metriken']['astra']['epochen_zeit'].append(epoche_zeit)
    
    astra_scheduler.step()
    
    if val_genauigkeit > astra_beste_val_genauigkeit:
        astra_beste_val_genauigkeit = val_genauigkeit
        astra_bester_modell_zustand = astra_modell.state_dict().copy()

    if (epoche + 1) % 5 == 0:
        print(f"ASTRA Epoche {epoche+1}/{epochen} - Val Genauigkeit: {val_genauigkeit:.4f}")

astra_modell.load_state_dict(astra_bester_modell_zustand)
print("ASTRA-Training abgeschlossen")

zeitmesser.phase_beenden()

# ==========================
# 5. ASTRA-Modell Evaluierung
# ==========================
zeitmesser.phase_starten("ASTRA-Modell Evaluierung")

astra_modell.eval()
astra_alle_vorhersagen = []
astra_alle_labels = []
astra_alle_konfidenzen = []
astra_inferenz_zeiten = []

with torch.no_grad():
    for bilder, labels in astra_test_loader:
        bilder, labels = bilder.to(device), labels.to(device)
        
        start_zeit = time.time()
        ausgaben = astra_modell(bilder)
        end_zeit = time.time()
        
        batch_inferenz_zeit = (end_zeit - start_zeit) / bilder.size(0)
        astra_inferenz_zeiten.extend([batch_inferenz_zeit] * bilder.size(0))
        
        wahrscheinlichkeiten = F.softmax(ausgaben, dim=1)
        konfidenzen, vorhersagen = torch.max(wahrscheinlichkeiten, 1)
        
        astra_alle_vorhersagen.extend(vorhersagen.cpu().numpy())
        astra_alle_labels.extend(labels.cpu().numpy())
        astra_alle_konfidenzen.extend(konfidenzen.cpu().numpy())

# Metriken berechnen
astra_test_genauigkeit = sum(np.array(astra_alle_vorhersagen) == np.array(astra_alle_labels)) / len(astra_alle_labels)
astra_praezision, astra_sensitivitaet, astra_f1, _ = precision_recall_fscore_support(
    astra_alle_labels, astra_alle_vorhersagen, average='weighted', zero_division=0)

# Test-Metriken sammeln
daten_sammler.alle_daten['test_metriken']['astra'] = {
    'testgenauigkeit': astra_test_genauigkeit,
    'beste_val_genauigkeit': astra_beste_val_genauigkeit,
    'praezision': astra_praezision,
    'sensitivitaet': astra_sensitivitaet,
    'f1_score': astra_f1,
    'durchschnittliche_konfidenz': np.mean(astra_alle_konfidenzen),
    'durchschnittliche_inferenzzeit_ms': np.mean(astra_inferenz_zeiten) * 1000,
    'latenz_p95_ms': np.percentile(astra_inferenz_zeiten, 95) * 1000,
    'latenz_p99_ms': np.percentile(astra_inferenz_zeiten, 99) * 1000,
    'durchsatz_bilder_pro_sekunde': 1000 / (np.mean(astra_inferenz_zeiten) * 1000)
}

# Konfusionsmatrix sammeln
astra_km = confusion_matrix(astra_alle_labels, astra_alle_vorhersagen)
daten_sammler.alle_daten['konfusionsmatrizen']['astra'] = astra_km.tolist()

# Klassifikationsbericht sammeln
astra_bericht = classification_report(astra_alle_labels, astra_alle_vorhersagen, 
                                    target_names=[f'klasse_{i}' for i in range(astra_anzahl_klassen)], 
                                    output_dict=True, zero_division=0)
daten_sammler.alle_daten['klassifikationsberichte']['astra'] = astra_bericht

print(f"ASTRA Testgenauigkeit: {astra_test_genauigkeit:.4f}")

# Modell speichern
astra_modell_pfad = os.path.join(os.path.expanduser("~"), "Desktop", "verkehrszeichen_cnn_astra.pth")
torch.save(astra_bester_modell_zustand, astra_modell_pfad)

zeitmesser.phase_beenden()

# ==========================
# 6. GTSRB-Modell Training
# ==========================
zeitmesser.phase_starten("GTSRB-Modell Training")

print("GTSRB-Modell Training gestartet...")

gtsrb_train_loader = DataLoader(gtsrb_train, batch_size=batch_groesse, shuffle=True)
gtsrb_val_loader = DataLoader(gtsrb_val, batch_size=batch_groesse)
gtsrb_test_loader = DataLoader(gtsrb_test, batch_size=batch_groesse)

gtsrb_modell = VerkehrszeichenCNN(gtsrb_anzahl_klassen).to(device)
gtsrb_verlustfunktion = nn.CrossEntropyLoss()
gtsrb_optimierer = optim.Adam(gtsrb_modell.parameters(), lr=0.001, weight_decay=1e-4)
gtsrb_scheduler = optim.lr_scheduler.StepLR(gtsrb_optimierer, step_size=5, gamma=0.7)

gtsrb_beste_val_genauigkeit = 0.0

# Training
for epoche in range(epochen):
    epoche_start_zeit = time.time()
    
    # Training
    gtsrb_modell.train()
    laufender_verlust, korrekt = 0.0, 0
    for bilder, labels in gtsrb_train_loader:
        bilder, labels = bilder.to(device), labels.to(device)
        gtsrb_optimierer.zero_grad()
        ausgaben = gtsrb_modell(bilder)
        verlust = gtsrb_verlustfunktion(ausgaben, labels)
        verlust.backward()
        gtsrb_optimierer.step()
        laufender_verlust += verlust.item() * bilder.size(0)
        korrekt += (ausgaben.argmax(1) == labels).sum().item()

    train_verlust = laufender_verlust / len(gtsrb_train)
    train_genauigkeit = korrekt / len(gtsrb_train)

    # Validierung
    gtsrb_modell.eval()
    val_verlust, val_korrekt = 0.0, 0
    with torch.no_grad():
        for bilder, labels in gtsrb_val_loader:
            bilder, labels = bilder.to(device), labels.to(device)
            ausgaben = gtsrb_modell(bilder)
            verlust = gtsrb_verlustfunktion(ausgaben, labels)
            val_verlust += verlust.item() * bilder.size(0)
            val_korrekt += (ausgaben.argmax(1) == labels).sum().item()

    val_verlust /= len(gtsrb_val)
    val_genauigkeit = val_korrekt / len(gtsrb_val)
    epoche_zeit = time.time() - epoche_start_zeit
    
    # Daten sammeln
    daten_sammler.alle_daten['training_metriken']['gtsrb']['epochen'].append(epoche + 1)
    daten_sammler.alle_daten['training_metriken']['gtsrb']['train_verlust'].append(train_verlust)
    daten_sammler.alle_daten['training_metriken']['gtsrb']['train_genauigkeit'].append(train_genauigkeit)
    daten_sammler.alle_daten['training_metriken']['gtsrb']['val_verlust'].append(val_verlust)
    daten_sammler.alle_daten['training_metriken']['gtsrb']['val_genauigkeit'].append(val_genauigkeit)
    daten_sammler.alle_daten['training_metriken']['gtsrb']['epochen_zeit'].append(epoche_zeit)
    
    gtsrb_scheduler.step()
    
    if val_genauigkeit > gtsrb_beste_val_genauigkeit:
        gtsrb_beste_val_genauigkeit = val_genauigkeit
        gtsrb_bester_modell_zustand = gtsrb_modell.state_dict().copy()

    if (epoche + 1) % 5 == 0:
        print(f"GTSRB Epoche {epoche+1}/{epochen} - Val Genauigkeit: {val_genauigkeit:.4f}")

gtsrb_modell.load_state_dict(gtsrb_bester_modell_zustand)
print("GTSRB-Training abgeschlossen")

zeitmesser.phase_beenden()

# ==========================
# 7. GTSRB-Modell Evaluierung
# ==========================
zeitmesser.phase_starten("GTSRB-Modell Evaluierung")

gtsrb_modell.eval()
gtsrb_alle_vorhersagen = []
gtsrb_alle_labels = []
gtsrb_alle_konfidenzen = []
gtsrb_inferenz_zeiten = []

with torch.no_grad():
    for bilder, labels in gtsrb_test_loader:
        bilder, labels = bilder.to(device), labels.to(device)
        
        start_zeit = time.time()
        ausgaben = gtsrb_modell(bilder)
        end_zeit = time.time()
        
        batch_inferenz_zeit = (end_zeit - start_zeit) / bilder.size(0)
        gtsrb_inferenz_zeiten.extend([batch_inferenz_zeit] * bilder.size(0))
        
        wahrscheinlichkeiten = F.softmax(ausgaben, dim=1)
        konfidenzen, vorhersagen = torch.max(wahrscheinlichkeiten, 1)
        
        gtsrb_alle_vorhersagen.extend(vorhersagen.cpu().numpy())
        gtsrb_alle_labels.extend(labels.cpu().numpy())
        gtsrb_alle_konfidenzen.extend(konfidenzen.cpu().numpy())

# Metriken berechnen
gtsrb_test_genauigkeit = sum(np.array(gtsrb_alle_vorhersagen) == np.array(gtsrb_alle_labels)) / len(gtsrb_alle_labels)
gtsrb_praezision, gtsrb_sensitivitaet, gtsrb_f1, _ = precision_recall_fscore_support(
    gtsrb_alle_labels, gtsrb_alle_vorhersagen, average='weighted', zero_division=0)

# Test-Metriken sammeln
daten_sammler.alle_daten['test_metriken']['gtsrb'] = {
    'testgenauigkeit': gtsrb_test_genauigkeit,
    'beste_val_genauigkeit': gtsrb_beste_val_genauigkeit,
    'praezision': gtsrb_praezision,
    'sensitivitaet': gtsrb_sensitivitaet,
    'f1_score': gtsrb_f1,
    'durchschnittliche_konfidenz': np.mean(gtsrb_alle_konfidenzen),
    'durchschnittliche_inferenzzeit_ms': np.mean(gtsrb_inferenz_zeiten) * 1000,
    'latenz_p95_ms': np.percentile(gtsrb_inferenz_zeiten, 95) * 1000,
    'latenz_p99_ms': np.percentile(gtsrb_inferenz_zeiten, 99) * 1000,
    'durchsatz_bilder_pro_sekunde': 1000 / (np.mean(gtsrb_inferenz_zeiten) * 1000)
}

# Konfusionsmatrix sammeln
gtsrb_km = confusion_matrix(gtsrb_alle_labels, gtsrb_alle_vorhersagen)
daten_sammler.alle_daten['konfusionsmatrizen']['gtsrb'] = gtsrb_km.tolist()

# Klassifikationsbericht sammeln
gtsrb_bericht = classification_report(gtsrb_alle_labels, gtsrb_alle_vorhersagen, 
                                    target_names=[f'klasse_{i}' for i in range(gtsrb_anzahl_klassen)], 
                                    output_dict=True, zero_division=0)
daten_sammler.alle_daten['klassifikationsberichte']['gtsrb'] = gtsrb_bericht

print(f"GTSRB Testgenauigkeit: {gtsrb_test_genauigkeit:.4f}")

# Modell speichern
gtsrb_modell_pfad = os.path.join(os.path.expanduser("~"), "Desktop", "verkehrszeichen_cnn_gtsrb.pth")
torch.save(gtsrb_bester_modell_zustand, gtsrb_modell_pfad)

zeitmesser.phase_beenden()

# ==========================
# 8. Leistungsbenchmark
# ==========================
zeitmesser.phase_starten("Leistungsbenchmark")

print("Leistungsbenchmark durchführen...")

# Benchmark-Einstellungen
anzahl_warmup = 50
anzahl_test = 1000
batch_groessen = [1, 8, 16, 32]

def modell_benchmarken(modell, modell_name):
    benchmark_ergebnisse = {}
    modell.eval()
    
    for batch_groesse in batch_groessen:
        dummy_eingabe = torch.randn(batch_groesse, 3, 64, 64).to(device)
        
        # Aufwärmphase
        with torch.no_grad():
            for _ in range(anzahl_warmup):
                _ = modell(dummy_eingabe)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Latenzmessung
        latenzen = []
        with torch.no_grad():
            for _ in range(anzahl_test):
                start_zeit = time.perf_counter()
                ausgaben = modell(dummy_eingabe)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_zeit = time.perf_counter()
                latenzen.append(end_zeit - start_zeit)
        
        # Statistiken berechnen
        durchschnittliche_latenz = np.mean(latenzen) * 1000  # ms
        min_latenz = np.min(latenzen) * 1000
        max_latenz = np.max(latenzen) * 1000
        p95_latenz = np.percentile(latenzen, 95) * 1000
        p99_latenz = np.percentile(latenzen, 99) * 1000
        durchsatz = batch_groesse / (durchschnittliche_latenz / 1000)
        
        benchmark_ergebnisse[f'batch_{batch_groesse}'] = {
            'durchschnittliche_latenz_ms': durchschnittliche_latenz,
            'min_latenz_ms': min_latenz,
            'max_latenz_ms': max_latenz,
            'p95_latenz_ms': p95_latenz,
            'p99_latenz_ms': p99_latenz,
            'durchsatz_bilder_pro_sekunde': durchsatz
        }
    
    return benchmark_ergebnisse

# Beide Modelle benchmarken
astra_benchmark = modell_benchmarken(astra_modell, "ASTRA")
gtsrb_benchmark = modell_benchmarken(gtsrb_modell, "GTSRB")

daten_sammler.alle_daten['leistungsmetriken']['astra'] = astra_benchmark
daten_sammler.alle_daten['leistungsmetriken']['gtsrb'] = gtsrb_benchmark

# Modellgrößenanalyse
def modellgroesse_analysieren(modell):
    gesamt_parameter = sum(p.numel() for p in modell.parameters())
    trainierbare_parameter = sum(p.numel() for p in modell.parameters() if p.requires_grad)
    parameter_groesse_mb = gesamt_parameter * 4 / (1024 * 1024)  # 4 Bytes pro float32
    
    return {
        'gesamt_parameter': gesamt_parameter,
        'trainierbare_parameter': trainierbare_parameter,
        'modellgroesse_mb': parameter_groesse_mb
    }

astra_groesse = modellgroesse_analysieren(astra_modell)
gtsrb_groesse = modellgroesse_analysieren(gtsrb_modell)

daten_sammler.alle_daten['leistungsmetriken']['astra'].update(astra_groesse)
daten_sammler.alle_daten['leistungsmetriken']['gtsrb'].update(gtsrb_groesse)

print("Leistungsbenchmark abgeschlossen")

zeitmesser.phase_beenden()

# ==========================
# 9. Abschluss und Datenexport
# ==========================
zeitmesser.phase_starten("Datenexport")

# Überwachung stoppen
monitor.ueberwachung_stoppen()
zeitmesser.stoppen()

# Ressourcenzusammenfassung berechnen und hinzufügen
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

# Excel-Datei erstellen
excel_pfad = os.path.join(os.path.expanduser("~"), "Desktop", "cnn_training_ergebnisse.xlsx")
daten_sammler.zu_excel_exportieren(excel_pfad)

zeitmesser.phase_beenden()

# ==========================
# ABSCHLUSSBERICHT
# ==========================
print("\n" + "="*80)
print("CNN TRAINING ABGESCHLOSSEN")
print("="*80)
print(f"Gesamtlaufzeit: {daten_sammler.alle_daten['zeitmessungen']['gesamtlaufzeit_minuten']:.1f} Minuten")
print()
print("ENDERGEBNISSE:")
print(f"ASTRA  - Testgenauigkeit: {astra_test_genauigkeit:.4f} | F1-Score: {astra_f1:.4f}")
print(f"GTSRB  - Testgenauigkeit: {gtsrb_test_genauigkeit:.4f} | F1-Score: {gtsrb_f1:.4f}")
print()
print("GESPEICHERTE DATEIEN:")
print(f"- Modelle: verkehrszeichen_cnn_astra.pth, verkehrszeichen_cnn_gtsrb.pth")
print(f"- Excel-Daten: {os.path.basename(excel_pfad)}")
print()
print("EXCEL-ARBEITSBLÄTTER:")
print("- Allgemeine_Info: Grundlegende Informationen und Zusammenfassungen")
print("- ASTRA_Training/GTSRB_Training: Epochenweise Trainingsmetriken")
print("- ASTRA_Test_Metriken/GTSRB_Test_Metriken: Finale Testleistung")
print("- ASTRA_Konfusionsmatrix/GTSRB_Konfusionsmatrix: Klassifikationsmatrizen")
print("- ASTRA_Klassifikation/GTSRB_Klassifikation: Detaillierte Klassifikationsberichte")
print("- ASTRA_Leistung/GTSRB_Leistung: Benchmark- und Leistungsmetriken")
print("- Ressourcennutzung: CPU/RAM/GPU-Verbrauch über Zeit")
print("- Zeitmessungen: Ausführungszeiten aller Phasen")
print("="*80)