import torch
import os
import yaml
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
import glob

def find_latest_experiment():
    """Findet das neueste Trainingsexperiment"""
    base_path = Path("gtsdb_yolo_training")
    if not base_path.exists():
        return None
    
    # Finde alle gtsdb_experiment* Ordner
    experiments = list(base_path.glob("gtsdb_experiment*"))
    if not experiments:
        return None
    
    # Sortiere nach Erstellungszeit (neuestes zuletzt)
    experiments.sort(key=lambda x: x.stat().st_mtime)
    latest = experiments[-1]
    
    weights_path = latest / "weights" / "best.pt"
    if weights_path.exists():
        print(f"✅ Gefunden: {weights_path}")
        return str(weights_path)
    
    return None

def fix_yaml_paths(yaml_path):
    """Korrigiert Pfade in der YAML Datei für bessere Kompatibilität"""
    
    # Konvertiere zu absolutem Pfad
    yaml_path = Path(yaml_path).resolve()
    
    if not yaml_path.exists():
        print(f"❌ YAML Datei nicht gefunden: {yaml_path}")
        return None
    
    # Lade und korrigiere die YAML
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Verwende absolute Pfade
    base_path = yaml_path.parent
    config['path'] = str(base_path)
    config['train'] = 'images/train'
    config['val'] = 'images/val'
    config['test'] = 'images/test'
    
    # Neue YAML erstellen
    fixed_yaml_path = base_path / "gtsdb_fixed.yaml"
    with open(fixed_yaml_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"✅ Korrigierte YAML erstellt: {fixed_yaml_path}")
    return str(fixed_yaml_path)

def evaluate_existing_model():
    """Evaluiert das bereits trainierte Modell"""
    
    print("🔍 Suche nach trainiertem Modell...")
    
    # Finde das Modell
    weights_path = find_latest_experiment()
    if not weights_path:
        print("❌ Kein trainiertes Modell gefunden!")
        return None
    
    # Finde und korrigiere YAML
    yaml_path = fix_yaml_paths(r'C:\Users\timau\Desktop\gtsdb_yolo_dataset\gtsdb.yaml')
    if not yaml_path:
        return None
    
    print(f"\n=== EVALUATION MIT KORRIGIERTEN PFADEN ===")
    print(f"Modell: {weights_path}")
    print(f"YAML: {yaml_path}")
    
    # Wechsle ins YOLOv5 Verzeichnis
    original_dir = os.getcwd()
    os.chdir('yolov5')
    
    try:
        # Konvertiere Pfade für Windows
        weights_abs = Path(weights_path).resolve()
        yaml_abs = Path(yaml_path).resolve()
        
        # Validation
        print("\n📊 Führe Validation aus...")
        val_cmd = f'''python val.py --data "{yaml_abs}" --weights "{weights_abs}" --task val --project ../gtsdb_evaluation_fixed --name validation --save-txt --save-conf --device 0'''
        
        print(f"Command: {val_cmd}")
        result = os.system(val_cmd)
        
        if result == 0:
            print("✅ Validation erfolgreich!")
        else:
            print("⚠️ Validation hatte Probleme")
        
        # Test
        print("\n🧪 Führe Test aus...")
        test_cmd = f'''python val.py --data "{yaml_abs}" --weights "{weights_abs}" --task test --project ../gtsdb_evaluation_fixed --name test --save-txt --save-conf --device 0'''
        
        result = os.system(test_cmd)
        
        if result == 0:
            print("✅ Test erfolgreich!")
        else:
            print("⚠️ Test hatte Probleme")
            
    finally:
        os.chdir(original_dir)
    
    return weights_path, yaml_path

def run_inference_fixed(weights_path, yaml_path):
    """Führt Inferenz mit korrigierten Pfaden aus"""
    
    print(f"\n=== INFERENZ MIT KORRIGIERTEN PFADEN ===")
    
    original_dir = os.getcwd()
    os.chdir('yolov5')
    
    try:
        # Absolute Pfade
        weights_abs = Path(weights_path).resolve()
        test_images_abs = Path(r'C:\Users\timau\Desktop\gtsdb_yolo_dataset\images\test').resolve()
        
        if not test_images_abs.exists():
            print(f"❌ Test Images Ordner nicht gefunden: {test_images_abs}")
            return
        
        print(f"📂 Test Images: {test_images_abs}")
        
        # Inferenz Command
        inference_cmd = f'''python detect.py --weights "{weights_abs}" --source "{test_images_abs}" --project ../gtsdb_inference_fixed --name examples --save-txt --save-conf --line-thickness 2 --device 0'''
        
        print("🔍 Führe Inferenz aus...")
        result = os.system(inference_cmd)
        
        if result == 0:
            print("✅ Inferenz erfolgreich!")
            print("📁 Ergebnisse in: gtsdb_inference_fixed/examples/")
        else:
            print("⚠️ Inferenz hatte Probleme")
            
    finally:
        os.chdir(original_dir)

def copy_model_to_desktop(weights_path):
    """Kopiert das Modell auf den Desktop mit Fehlerbehandlung"""
    
    if not weights_path or not Path(weights_path).exists():
        print(f"❌ Modell nicht gefunden: {weights_path}")
        return
    
    try:
        desktop_model_path = r'C:\Users\timau\Desktop\gtsdb_yolov5_best.pt'
        shutil.copy2(weights_path, desktop_model_path)
        print(f"✅ Modell kopiert nach: {desktop_model_path}")
        
        # Zeige Modell-Info
        model_size = Path(weights_path).stat().st_size / (1024*1024)  # MB
        print(f"📊 Modell-Größe: {model_size:.1f} MB")
        
    except Exception as e:
        print(f"❌ Fehler beim Kopieren: {e}")

def show_training_results():
    """Zeigt die Trainingsergebnisse an"""
    
    print(f"\n=== TRAINING ZUSAMMENFASSUNG ===")
    print("📈 Erreichte Metriken:")
    print("   • Precision: 95.6%")  
    print("   • Recall: 90.9%")
    print("   • mAP@0.5: 95.5%")
    print("   • mAP@0.5:0.95: 71.3%")
    print("   • Trainingszeit: 9.1 Minuten")
    print("   • Epochs: 50")
    
    print(f"\n🎯 Interpretation:")
    print("   • Exzellente Performance für Traffic Sign Detection")
    print("   • Sehr hohe Precision (wenige False Positives)")
    print("   • Gute Recall (wenige verpasste Schilder)")
    print("   • mAP50 von 95.5% ist ausgezeichnet")

def main_fix():
    """Hauptfunktion zum Beheben der Probleme"""
    
    print("🔧 YOLOv5 ROBUSTE PROBLEM-FIX PIPELINE")
    print("="*50)
    
    # 1. Versuche automatische Suche
    print("Schritt 1: Automatische Modellsuche")
    weights_path = find_latest_experiment()
    
    # 2. Falls nicht gefunden, manuelle Suche
    if not weights_path:
        print("\nSchritt 1b: Manuelle Modellsuche")
        weights_path = manual_model_search()
    
    if weights_path:
        print(f"\n✅ MODELL GEFUNDEN: {weights_path}")
        
        # 3. Kopiere Modell an zugänglichen Ort
        print("\nSchritt 2: Modell-Kopie erstellen")
        accessible_model = copy_model_to_accessible_location(weights_path)
        
        # 4. Teste das Modell
        print("\nSchritt 3: Modell-Test")
        model_works = simple_inference_test(accessible_model or weights_path)
        
        # 5. Erstelle einfache Konfiguration
        print("\nSchritt 4: Konfiguration erstellen")
        yaml_path = create_simple_yaml()
        
        # 6. Zeige Zusammenfassung
        print("\nSchritt 5: Zusammenfassung")
        show_training_results()
        
        if accessible_model and model_works and yaml_path:
            print(f"\n🎉 ALLE PROBLEME BEHOBEN!")
            print(f"✅ Funktionsfähiges Modell: {accessible_model}")
            print(f"✅ Konfiguration: {yaml_path}")
            print(f"\n💡 NÄCHSTE SCHRITTE:")
            print(f"   1. Verwende das Modell: {accessible_model}")
            print(f"   2. Teste mit eigenen Bildern")
            print(f"   3. Integriere in deine Anwendung")
            
            return accessible_model, yaml_path
        else:
            print(f"\n⚠️ TEILWEISE ERFOLG")
            print(f"   • Modell verfügbar: {bool(accessible_model)}")
            print(f"   • Modell funktioniert: {model_works}")
            print(f"   • Konfiguration erstellt: {bool(yaml_path)}")
            
            return weights_path, yaml_path
            
    else:
        print("\n❌ KEIN MODELL GEFUNDEN")
        print("💡 Mögliche Lösungen:")
        print("   1. Training erneut ausführen")
        print("   2. Modell manuell suchen")
        print("   3. Pfade im ursprünglichen Code korrigieren")
        
        return None, None

if __name__ == "__main__":
    main_fix()