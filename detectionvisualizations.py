import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pandas as pd
import os
import random
import numpy as np

# Pfade
test_images_path = r"C:\Users\timau\Desktop\Datensaetze\GTSDB\Test"
predictions_file = r"G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results\gtsdb_pure_yolo_predictions.txt"
output_save_path = r"G:\Meine Ablage\Maturaarbeit\Resultate\Detection"

def load_predictions(file_path):
    """Lade YOLO-Vorhersagen aus der TXT-Datei"""
    predictions = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split(',')
            if len(parts) >= 10:
                filename = parts[0]
                try:
                    x1, y1, x2, y2 = map(int, parts[1:5])
                    confidence = float(parts[5])
                    img_width, img_height = map(int, parts[8:10])
                except (ValueError, IndexError) as e:
                    print(f"Warnung: Ungültiges Format in Zeile übersprungen: {line.strip()} - Fehler: {e}")
                    continue
                
                if filename not in predictions:
                    predictions[filename] = []
                
                predictions[filename].append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'img_size': (img_width, img_height)
                })
    
    return predictions

def visualize_predictions(predictions, images_path, output_path):
    """Visualisiere Vorhersagen auf allen Bildern und speichere sie einzeln"""
    
    # Stelle sicher, dass der Ausgabeordner existiert
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Ausgabeordner erstellt: {output_path}")

    # Verarbeite jedes Bild mit Vorhersagen
    for image_name, preds in predictions.items():
        image_path = os.path.join(images_path, image_name)
        
        # Versuche verschiedene Dateierweiterungen
        if not os.path.exists(image_path):
            base_name = os.path.splitext(image_name)[0]
            image_path = os.path.join(images_path, base_name + '.ppm')
        
        if not os.path.exists(image_path):
            print(f"Bild nicht gefunden: {image_name}")
            continue
        
        try:
            img = Image.open(image_path)
            fig, ax = plt.subplots(1, figsize=(10, 6))
            ax.imshow(img)
            
            # Zeichne Bounding Boxes
            for pred in preds:
                x1, y1, x2, y2 = pred['bbox']
                confidence = pred['confidence']
                
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                ax.text(
                    x1, y1-5,
                    f'{confidence:.3f}',
                    color='red',
                    fontsize=8,
                    fontweight='bold',
                )
            
            ax.set_title(f'Detektionen für {image_name}', fontsize=12)
            ax.axis('off')
            
            # Speichere die Visualisierung
            output_file = os.path.join(output_path, f"detected_{os.path.splitext(image_name)[0]}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig) # Schließe das Plot-Fenster, um Speicher zu sparen
            print(f"Visualisierung gespeichert: {output_file}")
            
        except Exception as e:
            print(f"Fehler beim Verarbeiten von {image_name}: {e}")
    
    print("\nVisualisierung aller Bilder abgeschlossen.")

def print_statistics(predictions):
    """Zeige Statistiken zu den Vorhersagen"""
    total_images = len(predictions)
    total_detections = sum(len(preds) for preds in predictions.values())
    avg_detections = total_detections / total_images if total_images > 0 else 0
    
    all_confidences = []
    for preds in predictions.values():
        for pred in preds:
            all_confidences.append(pred['confidence'])
    
    print(f"\n=== VORHERSAGE STATISTIKEN ===")
    print(f"Bilder mit Detektionen: {total_images}")
    print(f"Gesamt Detektionen: {total_detections}")
    print(f"Durchschnitt pro Bild: {avg_detections:.2f}")
    
    if all_confidences:
        print(f"Mittlere Konfidenz: {np.mean(all_confidences):.3f}")
        print(f"Median Konfidenz: {np.median(all_confidences):.3f}")
        print(f"Min/Max Konfidenz: {np.min(all_confidences):.3f} / {np.max(all_confidences):.3f}")

# Hauptprogramm
if __name__ == "__main__":
    print("Lade YOLO-Vorhersagen...")
    predictions = load_predictions(predictions_file)
    
    print_statistics(predictions)
    
    print(f"\nErstelle Visualisierungen für alle {len(predictions)} Bilder und speichere sie in {output_save_path}...")
    visualize_predictions(predictions, test_images_path, output_save_path)
    
    print("Fertig!")