import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# CSV-Datei laden - Wähle eine der folgenden Optionen:

# Option 1: Raw String (empfohlen)
df = pd.read_csv(r'C:\Users\timau\Desktop\gtsdb_detection6\results.csv')

# Option 2: Forward Slashes (funktioniert auch unter Windows)
# df = pd.read_csv('C:/Users/timau/Desktop/gtsdb_detection6/results.csv')

# Option 3: Escaped Backslashes
# df = pd.read_csv('C:\\Users\\timau\\Desktop\\gtsdb_detection6\\results.csv')

# Datenvorverarbeitung
df['time_minutes'] = df['time'] / 60  # Zeit in Minuten
df['mAP50_percent'] = df['metrics/mAP50(B)'] * 100
df['mAP50_95_percent'] = df['metrics/mAP50-95(B)'] * 100
df['precision_percent'] = df['metrics/precision(B)'] * 100
df['recall_percent'] = df['metrics/recall(B)'] * 100

# Farbpalette definieren
colors = plt.cm.Set2(np.linspace(0, 1, 8))

print("YOLO Training Analyse - Einzelne Plots werden erstellt...\n")

# =============================================================================
# PLOT 1: Hauptmetriken (Precision, Recall, mAP@50, mAP@50-95)
# =============================================================================
plt.figure(figsize=(12, 8))
plt.plot(df['epoch'], df['precision_percent'], 'o-', color=colors[0], linewidth=2, markersize=4, label='Precision')
plt.plot(df['epoch'], df['recall_percent'], 's-', color=colors[1], linewidth=2, markersize=4, label='Recall')
plt.plot(df['epoch'], df['mAP50_percent'], '^-', color=colors[2], linewidth=3, markersize=4, label='mAP@50')
plt.plot(df['epoch'], df['mAP50_95_percent'], 'd-', color=colors[3], linewidth=2, markersize=4, label='mAP@50-95')
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('Prozent (%)', fontsize=12)
plt.title('YOLO Training - Detektions-Metriken Verlauf', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)

# Finale Werte als Text hinzufügen
final_values = [
    f"Finale mAP@50: {df['mAP50_percent'].iloc[-1]:.1f}%",
    f"Finale Precision: {df['precision_percent'].iloc[-1]:.1f}%",
    f"Finale Recall: {df['recall_percent'].iloc[-1]:.1f}%",
    f"Finale mAP@50-95: {df['mAP50_95_percent'].iloc[-1]:.1f}%"
]
plt.text(0.02, 0.98, '\n'.join(final_values), transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
plt.tight_layout()
plt.show()

# =============================================================================
# PLOT 2: Training Losses
# =============================================================================
plt.figure(figsize=(12, 8))
plt.plot(df['epoch'], df['train/box_loss'], 'o-', color=colors[4], linewidth=2, markersize=4, label='Box Loss')
plt.plot(df['epoch'], df['train/cls_loss'], 's-', color=colors[5], linewidth=2, markersize=4, label='Class Loss')
plt.plot(df['epoch'], df['train/dfl_loss'], '^-', color=colors[6], linewidth=2, markersize=4, label='DFL Loss')
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('Loss Wert', fontsize=12)
plt.title('YOLO Training - Training Loss Verlauf', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# PLOT 3: Validation Losses
# =============================================================================
plt.figure(figsize=(12, 8))
plt.plot(df['epoch'], df['val/box_loss'], 'o-', color=colors[4], linewidth=2, markersize=4, label='Val Box Loss')
plt.plot(df['epoch'], df['val/cls_loss'], 's-', color=colors[5], linewidth=2, markersize=4, label='Val Class Loss')
plt.plot(df['epoch'], df['val/dfl_loss'], '^-', color=colors[6], linewidth=2, markersize=4, label='Val DFL Loss')
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('Loss Wert', fontsize=12)
plt.title('YOLO Training - Validation Loss Verlauf', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# PLOT 4: Learning Rate Schedule
# =============================================================================
plt.figure(figsize=(12, 8))
plt.plot(df['epoch'], df['lr/pg0'], 'o-', color=colors[0], linewidth=2, markersize=4, label='LR PG0')
plt.plot(df['epoch'], df['lr/pg1'], 's-', color=colors[1], linewidth=2, markersize=4, label='LR PG1') 
plt.plot(df['epoch'], df['lr/pg2'], '^-', color=colors[2], linewidth=2, markersize=4, label='LR PG2')
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('YOLO Training - Learning Rate Schedule', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Logarithmische Skala für bessere Sichtbarkeit
plt.tight_layout()
plt.show()

# =============================================================================
# PLOT 5: Training Zeit
# =============================================================================
plt.figure(figsize=(12, 8))
plt.plot(df['epoch'], df['time_minutes'], 'o-', color=colors[7], linewidth=3, markersize=5, label='Kumulative Zeit')
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('Zeit (Minuten)', fontsize=12)
plt.title('YOLO Training - Kumulative Training Zeit', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Zeit pro Epoche berechnen und anzeigen
time_per_epoch = np.diff(np.concatenate([[0], df['time_minutes'].values]))
avg_time_per_epoch = np.mean(time_per_epoch)
plt.text(0.02, 0.98, f'Ø Zeit/Epoche: {avg_time_per_epoch:.1f} min\nGesamt: {df["time_minutes"].iloc[-1]:.0f} min', 
         transform=plt.gca().transAxes, verticalalignment='top', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
plt.tight_layout()
plt.show()

# =============================================================================
# PLOT 6: Precision vs Recall Scatter
# =============================================================================
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['recall_percent'], df['precision_percent'], 
                     c=df['epoch'], cmap='viridis', s=60, alpha=0.7, edgecolors='black')
plt.xlabel('Recall (%)', fontsize=12)
plt.ylabel('Precision (%)', fontsize=12)
plt.title('YOLO Training - Precision vs Recall (Farbe = Epoche)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(0, 100)
plt.ylim(0, 100)

# Colorbar für Epochen
cbar = plt.colorbar(scatter)
cbar.set_label('Epoche', fontsize=11)

# Beste Performance markieren
best_map50_idx = df['mAP50_percent'].idxmax()
plt.scatter(df.loc[best_map50_idx, 'recall_percent'], 
           df.loc[best_map50_idx, 'precision_percent'], 
           s=200, color='red', marker='*', edgecolors='black', linewidths=2,
           label=f'Beste mAP@50 (Epoche {df.loc[best_map50_idx, "epoch"]})')
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

# =============================================================================
# PLOT 7: Training vs Validation Box Loss
# =============================================================================
plt.figure(figsize=(12, 8))
plt.plot(df['epoch'], df['train/box_loss'], 'o-', color='blue', linewidth=2, markersize=4, label='Training Box Loss')
plt.plot(df['epoch'], df['val/box_loss'], 's-', color='red', linewidth=2, markersize=4, label='Validation Box Loss')
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('Box Loss', fontsize=12)
plt.title('YOLO Training - Box Loss: Training vs Validation', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# PLOT 8: Training vs Validation Class Loss
# =============================================================================
plt.figure(figsize=(12, 8))
plt.plot(df['epoch'], df['train/cls_loss'], 'o-', color='green', linewidth=2, markersize=4, label='Training Class Loss')
plt.plot(df['epoch'], df['val/cls_loss'], 's-', color='orange', linewidth=2, markersize=4, label='Validation Class Loss')
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('Class Loss', fontsize=12)
plt.title('YOLO Training - Class Loss: Training vs Validation', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# PLOT 9: Training vs Validation DFL Loss
# =============================================================================
plt.figure(figsize=(12, 8))
plt.plot(df['epoch'], df['train/dfl_loss'], 'o-', color='purple', linewidth=2, markersize=4, label='Training DFL Loss')
plt.plot(df['epoch'], df['val/dfl_loss'], 's-', color='brown', linewidth=2, markersize=4, label='Validation DFL Loss')
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('DFL Loss', fontsize=12)
plt.title('YOLO Training - DFL Loss: Training vs Validation', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# PLOT 10: Gesamt Loss Vergleich
# =============================================================================
plt.figure(figsize=(12, 8))
total_train_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
total_val_loss = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']

plt.plot(df['epoch'], total_train_loss, 'o-', color='navy', linewidth=2, markersize=4, label='Training Gesamt Loss')
plt.plot(df['epoch'], total_val_loss, 's-', color='darkred', linewidth=2, markersize=4, label='Validation Gesamt Loss')
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('Gesamt Loss', fontsize=12)
plt.title('YOLO Training - Gesamt Loss: Training vs Validation', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# PLOT 11: Korrelations-Heatmap
# =============================================================================
plt.figure(figsize=(12, 10))
correlation_cols = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 
                   'metrics/mAP50-95(B)', 'train/box_loss', 'train/cls_loss', 
                   'val/box_loss', 'val/cls_loss', 'lr/pg0']
corr_matrix = df[correlation_cols].corr()

# Heatmap erstellen
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
           square=True, cbar_kws={'shrink': 0.8}, fmt='.2f')
plt.title('YOLO Training - Korrelationsmatrix der Metriken', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Statistiken ausgeben
print("\n" + "="*60)
print("YOLO TRAINING ANALYSE - ZUSAMMENFASSUNG")
print("="*60)
print(f"Anzahl Epochen: {len(df)}")
print(f"Gesamt Training Zeit: {df['time_minutes'].iloc[-1]:.1f} Minuten")
print(f"Durchschnittliche Zeit pro Epoche: {avg_time_per_epoch:.1f} Minuten")
print(f"\nFINALE PERFORMANCE:")
print(f"  • mAP@50: {df['mAP50_percent'].iloc[-1]:.2f}%")
print(f"  • mAP@50-95: {df['mAP50_95_percent'].iloc[-1]:.2f}%")
print(f"  • Precision: {df['precision_percent'].iloc[-1]:.2f}%")
print(f"  • Recall: {df['recall_percent'].iloc[-1]:.2f}%")
print(f"\nBESTE ERGEBNISSE:")
print(f"  • Beste mAP@50: {df['mAP50_percent'].max():.2f}% (Epoche {df.loc[df['mAP50_percent'].idxmax(), 'epoch']})")
print(f"  • Beste mAP@50-95: {df['mAP50_95_percent'].max():.2f}% (Epoche {df.loc[df['mAP50_95_percent'].idxmax(), 'epoch']})")
print(f"  • Beste Precision: {df['precision_percent'].max():.2f}% (Epoche {df.loc[df['precision_percent'].idxmax(), 'epoch']})")
print(f"  • Beste Recall: {df['recall_percent'].max():.2f}% (Epoche {df.loc[df['recall_percent'].idxmax(), 'epoch']})")
print(f"\nLOSS REDUKTION:")
print(f"  • Training Box Loss: {df['train/box_loss'].iloc[0]:.3f} → {df['train/box_loss'].iloc[-1]:.3f}")
print(f"  • Training Class Loss: {df['train/cls_loss'].iloc[0]:.3f} → {df['train/cls_loss'].iloc[-1]:.3f}")
print(f"  • Validation Box Loss: {df['val/box_loss'].iloc[0]:.3f} → {df['val/box_loss'].iloc[-1]:.3f}")
print(f"  • Validation Class Loss: {df['val/cls_loss'].iloc[0]:.3f} → {df['val/cls_loss'].iloc[-1]:.3f}")
print("="*60)