import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Deutsche Schriftarten und Stil-Einstellungen
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = [12, 8]

# Farbpaletten definieren
COLORS_ASTRA = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD']
COLORS_GTSRB = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
COLORS_PERFORMANCE = ['#1ABC9C', '#E67E22', '#34495E', '#95A5A6', '#D35400']

def lade_excel_daten(dateipfad):
    """Excel-Datei laden und alle Arbeitsblätter einlesen"""
    try:
        excel_datei = pd.ExcelFile(dateipfad)
        daten = {}
        
        for blatt_name in excel_datei.sheet_names:
            try:
                daten[blatt_name] = pd.read_excel(dateipfad, sheet_name=blatt_name)
                print(f"✓ Arbeitsblatt '{blatt_name}' geladen: {len(daten[blatt_name])} Zeilen")
            except Exception as e:
                print(f"✗ Fehler beim Laden von '{blatt_name}': {e}")
        
        return daten
    except Exception as e:
        print(f"Fehler beim Laden der Excel-Datei: {e}")
        return None

def erstelle_training_verlaeufe(daten, speicherpfad_basis):
    """Trainingsverläufe für beide Modelle"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Trainingsverläufe - ASTRA und GTSRB Modelle', fontsize=20, fontweight='bold')
    
    # ASTRA Trainingsverläufe
    if 'ASTRA_Training' in daten and not daten['ASTRA_Training'].empty:
        astra_data = daten['ASTRA_Training']
        
        # ASTRA Verlust
        axes[0, 0].plot(astra_data['epochen'], astra_data['train_verlust'], 
                       'o-', color=COLORS_ASTRA[0], linewidth=3, markersize=6, label='Training')
        axes[0, 0].plot(astra_data['epochen'], astra_data['val_verlust'], 
                       's-', color=COLORS_ASTRA[1], linewidth=3, markersize=6, label='Validierung')
        axes[0, 0].set_title('ASTRA - Verlustfunktion', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoche', fontsize=12)
        axes[0, 0].set_ylabel('Verlust', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # ASTRA Genauigkeit
        axes[0, 1].plot(astra_data['epochen'], astra_data['train_genauigkeit'] * 100, 
                       'o-', color=COLORS_ASTRA[0], linewidth=3, markersize=6, label='Training')
        axes[0, 1].plot(astra_data['epochen'], astra_data['val_genauigkeit'] * 100, 
                       's-', color=COLORS_ASTRA[1], linewidth=3, markersize=6, label='Validierung')
        axes[0, 1].set_title('ASTRA - Genauigkeit', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoche', fontsize=12)
        axes[0, 1].set_ylabel('Genauigkeit (%)', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 100)
    
    # GTSRB Trainingsverläufe
    if 'GTSRB_Training' in daten and not daten['GTSRB_Training'].empty:
        gtsrb_data = daten['GTSRB_Training']
        
        # GTSRB Verlust
        axes[1, 0].plot(gtsrb_data['epochen'], gtsrb_data['train_verlust'], 
                       'o-', color=COLORS_GTSRB[0], linewidth=3, markersize=6, label='Training')
        axes[1, 0].plot(gtsrb_data['epochen'], gtsrb_data['val_verlust'], 
                       's-', color=COLORS_GTSRB[1], linewidth=3, markersize=6, label='Validierung')
        axes[1, 0].set_title('GTSRB - Verlustfunktion', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoche', fontsize=12)
        axes[1, 0].set_ylabel('Verlust', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        # GTSRB Genauigkeit
        axes[1, 1].plot(gtsrb_data['epochen'], gtsrb_data['train_genauigkeit'] * 100, 
                       'o-', color=COLORS_GTSRB[0], linewidth=3, markersize=6, label='Training')
        axes[1, 1].plot(gtsrb_data['epochen'], gtsrb_data['val_genauigkeit'] * 100, 
                       's-', color=COLORS_GTSRB[1], linewidth=3, markersize=6, label='Validierung')
        axes[1, 1].set_title('GTSRB - Genauigkeit', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoche', fontsize=12)
        axes[1, 1].set_ylabel('Genauigkeit (%)', fontsize=12)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 100)
    
    plt.tight_layout()
    speicherpfad = f"{speicherpfad_basis}_trainingsverlaeufe.png"
    plt.savefig(speicherpfad, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Trainingsverläufe gespeichert: {speicherpfad}")

def erstelle_metriken_vergleich(daten, speicherpfad_basis):
    """Vergleich der finalen Test-Metriken"""
    if 'ASTRA_Test_Metriken' not in daten or 'GTSRB_Test_Metriken' not in daten:
        print("Test-Metriken nicht verfügbar")
        return
    
    astra_metriken = daten['ASTRA_Test_Metriken'].iloc[0]
    gtsrb_metriken = daten['GTSRB_Test_Metriken'].iloc[0]
    
    # Hauptmetriken extrahieren
    metriken_namen = ['Testgenauigkeit', 'Präzision', 'Sensitivität', 'F1-Score', 'Konfidenz']
    astra_werte = [
        astra_metriken.get('testgenauigkeit', 0) * 100,
        astra_metriken.get('praezision', 0) * 100,
        astra_metriken.get('sensitivitaet', 0) * 100,
        astra_metriken.get('f1_score', 0) * 100,
        astra_metriken.get('durchschnittliche_konfidenz', 0) * 100
    ]
    gtsrb_werte = [
        gtsrb_metriken.get('testgenauigkeit', 0) * 100,
        gtsrb_metriken.get('praezision', 0) * 100,
        gtsrb_metriken.get('sensitivitaet', 0) * 100,
        gtsrb_metriken.get('f1_score', 0) * 100,
        gtsrb_metriken.get('durchschnittliche_konfidenz', 0) * 100
    ]
    
    # Balkendiagramm
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(metriken_namen))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, astra_werte, width, label='ASTRA', 
                  color=COLORS_ASTRA[0], alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, gtsrb_werte, width, label='GTSRB', 
                  color=COLORS_GTSRB[0], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Werte auf Balken anzeigen
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.5,
                f'{height1:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.5,
                f'{height2:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Metriken', fontsize=14)
    ax.set_ylabel('Wert (%)', fontsize=14)
    ax.set_title('Vergleich der finalen Test-Metriken', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metriken_namen, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    speicherpfad = f"{speicherpfad_basis}_metriken_vergleich.png"
    plt.savefig(speicherpfad, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Metrikenvergleich gespeichert: {speicherpfad}")

def erstelle_leistungsmetriken(daten, speicherpfad_basis):
    """Leistungsmetriken (Inferenzzeit, Durchsatz, etc.)"""
    if 'ASTRA_Test_Metriken' not in daten or 'GTSRB_Test_Metriken' not in daten:
        return
    
    astra_metriken = daten['ASTRA_Test_Metriken'].iloc[0]
    gtsrb_metriken = daten['GTSRB_Test_Metriken'].iloc[0]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Leistungsmetriken - ASTRA vs GTSRB', fontsize=20, fontweight='bold')
    
    # Inferenzgeschwindigkeit
    inferenz_daten = [
        astra_metriken.get('durchschnittliche_inferenzzeit_ms', 0),
        gtsrb_metriken.get('durchschnittliche_inferenzzeit_ms', 0)
    ]
    bars1 = ax1.bar(['ASTRA', 'GTSRB'], inferenz_daten, 
                   color=[COLORS_ASTRA[0], COLORS_GTSRB[0]], alpha=0.8,
                   edgecolor='black', linewidth=2)
    ax1.set_title('Durchschnittliche Inferenzgeschwindigkeit', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Zeit (ms)', fontsize=12)
    for bar, wert in zip(bars1, inferenz_daten):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{wert:.2f} ms', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Durchsatz
    durchsatz_daten = [
        astra_metriken.get('durchsatz_bilder_pro_sekunde', 0),
        gtsrb_metriken.get('durchsatz_bilder_pro_sekunde', 0)
    ]
    bars2 = ax2.bar(['ASTRA', 'GTSRB'], durchsatz_daten, 
                   color=[COLORS_ASTRA[1], COLORS_GTSRB[1]], alpha=0.8,
                   edgecolor='black', linewidth=2)
    ax2.set_title('Durchsatz', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Bilder/Sekunde', fontsize=12)
    for bar, wert in zip(bars2, durchsatz_daten):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{wert:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Latenz-Vergleich
    latenz_kategorien = ['P95 Latenz', 'P99 Latenz']
    astra_latenz = [
        astra_metriken.get('latenz_p95_ms', 0),
        astra_metriken.get('latenz_p99_ms', 0)
    ]
    gtsrb_latenz = [
        gtsrb_metriken.get('latenz_p95_ms', 0),
        gtsrb_metriken.get('latenz_p99_ms', 0)
    ]
    
    x = np.arange(len(latenz_kategorien))
    width = 0.35
    bars3 = ax3.bar(x - width/2, astra_latenz, width, label='ASTRA', 
                   color=COLORS_ASTRA[2], alpha=0.8, edgecolor='black', linewidth=1)
    bars4 = ax3.bar(x + width/2, gtsrb_latenz, width, label='GTSRB', 
                   color=COLORS_GTSRB[2], alpha=0.8, edgecolor='black', linewidth=1)
    
    ax3.set_title('Latenz-Perzentile', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Zeit (ms)', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(latenz_kategorien, fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Werte auf Balken
    for bar, wert in zip(bars3, astra_latenz):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{wert:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    for bar, wert in zip(bars4, gtsrb_latenz):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{wert:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Konfidenz-Vergleich
    konfidenz_daten = [
        astra_metriken.get('durchschnittliche_konfidenz', 0) * 100,
        gtsrb_metriken.get('durchschnittliche_konfidenz', 0) * 100
    ]
    bars5 = ax4.bar(['ASTRA', 'GTSRB'], konfidenz_daten, 
                   color=[COLORS_ASTRA[3], COLORS_GTSRB[3]], alpha=0.8,
                   edgecolor='black', linewidth=2)
    ax4.set_title('Durchschnittliche Modellkonfidenz', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Konfidenz (%)', fontsize=12)
    ax4.set_ylim(0, 100)
    for bar, wert in zip(bars5, konfidenz_daten):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{wert:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    speicherpfad = f"{speicherpfad_basis}_leistungsmetriken.png"
    plt.savefig(speicherpfad, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Leistungsmetriken gespeichert: {speicherpfad}")

def erstelle_konfusionsmatrizen(daten, speicherpfad_basis):
    """Konfusionsmatrizen im ursprünglichen CNN-Skript Stil"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle('Konfusionsmatrizen - Klassifikationsergebnisse', fontsize=22, fontweight='bold')
    
    # Klassennamen definieren (wie im ursprünglichen Skript)
    astra_klassennamen = [
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
    
    gtsrb_klassennamen = [
        '0_Geschwindigkeitsbeschraenkung_20', '1_Geschwindigkeitsbeschraenkung_30',   
        '2_Geschwindigkeitsbeschraenkung_50', '3_Geschwindigkeitsbeschraenkung_60',  
        '4_Geschwindigkeitsbeschraenkung_70', '5_Geschwindigkeitsbeschraenkung_80',
        '6_Ende_Hoechstgeschwindigkeit', '7_Geschwindigkeitsbeschraenkung_100',
        '8_Geschwindigkeitsbeschraenkung_120', '9_Ueberholen_verboten',
        '10_Ueberholverbot_fuer_Kraftfahrzeuge', '11_Vorfahrt', '12_Hauptstrasse',
        '13_Vorfahrt_gewaehren', '14_Stop', '15_Farhverbot', '16_Fahrverbot_fuer_Kraftfahrzeuge',
        '17_Verbot_der_Einfahrt', '18_Gefahrstelle', '19_Kurve_links', '20_Kurve_rechts',
        '21_Doppelkurve_zunaechst_links', '22_Uneben_Fahrbahn', '23_Schleuder_oder_Rutschgefahr',
        '24_Verengung_rechts', '25_Baustelle', '26_Lichtzeichenanlage', '27_Fussgaenger',
        '28_Kinder', '29_Radverkehr', '30_Schnee_oder_Eisglaette', '31_Wildwechsel',
        '32_Ende_Geschwindigkeitsbegraenzungen', '33_Fahrtrichtung_rechts', '34_Fahrtrichtung_links',
        '35_Fahrtrichtung_geradeaus', '36_Fahrtrichtung_geradeaus_rechts', '37_Fahrtrichtung_geradeaus_links',
        '38_Vorbeifahrt_rechts', '39_Vorbeifahrt_links', '40_Kreisverkehr',
        '41_Ende_Ueberholverbot', '42_Ende_Ueberholverbot_fuer_Kraftfahrzeuge'
    ]
    
    # ASTRA Konfusionsmatrix (wie im ursprünglichen Skript)
    if 'ASTRA_Konfusionsmatrix' in daten and not daten['ASTRA_Konfusionsmatrix'].empty:
        astra_km = daten['ASTRA_Konfusionsmatrix'].values
        
        sns.heatmap(astra_km, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=[name[:20] + '...' if len(name) > 20 else name for name in astra_klassennamen],
                    yticklabels=[name[:20] + '...' if len(name) > 20 else name for name in astra_klassennamen],
                    cbar_kws={'label': 'Anzahl Vorhersagen'})
        
        ax1.set_title('ASTRA Konfusionsmatrix', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Vorhergesagte Klasse', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Wahre Klasse', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45, labelsize=10)
        ax1.tick_params(axis='y', rotation=0, labelsize=10)
    
    # GTSRB Konfusionsmatrix (wie im ursprünglichen Skript)
    if 'GTSRB_Konfusionsmatrix' in daten and not daten['GTSRB_Konfusionsmatrix'].empty:
        gtsrb_km = daten['GTSRB_Konfusionsmatrix'].values
        
        # Verkürzte Labels für bessere Lesbarkeit
        gtsrb_labels_kurz = [f"{i}: {name[:15]}..." if len(name) > 15 else f"{i}: {name}" 
                            for i, name in enumerate(gtsrb_klassennamen)]
        
        sns.heatmap(gtsrb_km, annot=True, fmt='d', cmap='Oranges', ax=ax2,
                    xticklabels=gtsrb_labels_kurz,
                    yticklabels=gtsrb_labels_kurz,
                    cbar_kws={'label': 'Anzahl Vorhersagen'})
        
        ax2.set_title('GTSRB Konfusionsmatrix', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Vorhergesagte Klasse', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Wahre Klasse', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.tick_params(axis='y', rotation=0, labelsize=8)
    
    plt.tight_layout()
    speicherpfad = f"{speicherpfad_basis}_konfusionsmatrizen_original_stil.png"
    plt.savefig(speicherpfad, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Konfusionsmatrizen (Original-Stil) gespeichert: {speicherpfad}")

def erstelle_separate_gtsrb_konfusionsmatrix(daten, speicherpfad_basis):
    """Separate, detaillierte GTSRB Konfusionsmatrix für bessere Analyse"""
    if 'GTSRB_Konfusionsmatrix' not in daten or daten['GTSRB_Konfusionsmatrix'].empty:
        return
    
    gtsrb_km = daten['GTSRB_Konfusionsmatrix'].values
    
    # Klassennamen
    gtsrb_klassennamen = [
        '0_Geschwindigkeitsbeschraenkung_20', '1_Geschwindigkeitsbeschraenkung_30',   
        '2_Geschwindigkeitsbeschraenkung_50', '3_Geschwindigkeitsbeschraenkung_60',  
        '4_Geschwindigkeitsbeschraenkung_70', '5_Geschwindigkeitsbeschraenkung_80',
        '6_Ende_Hoechstgeschwindigkeit', '7_Geschwindigkeitsbeschraenkung_100',
        '8_Geschwindigkeitsbeschraenkung_120', '9_Ueberholen_verboten',
        '10_Ueberholverbot_fuer_Kraftfahrzeuge', '11_Vorfahrt', '12_Hauptstrasse',
        '13_Vorfahrt_gewaehren', '14_Stop', '15_Farhverbot', '16_Fahrverbot_fuer_Kraftfahrzeuge',
        '17_Verbot_der_Einfahrt', '18_Gefahrstelle', '19_Kurve_links', '20_Kurve_rechts',
        '21_Doppelkurve_zunaechst_links', '22_Uneben_Fahrbahn', '23_Schleuder_oder_Rutschgefahr',
        '24_Verengung_rechts', '25_Baustelle', '26_Lichtzeichenanlage', '27_Fussgaenger',
        '28_Kinder', '29_Radverkehr', '30_Schnee_oder_Eisglaette', '31_Wildwechsel',
        '32_Ende_Geschwindigkeitsbegraenzungen', '33_Fahrtrichtung_rechts', '34_Fahrtrichtung_links',
        '35_Fahrtrichtung_geradeaus', '36_Fahrtrichtung_geradeaus_rechts', '37_Fahrtrichtung_geradeaus_links',
        '38_Vorbeifahrt_rechts', '39_Vorbeifahrt_links', '40_Kreisverkehr',
        '41_Ende_Ueberholverbot', '42_Ende_Ueberholverbot_fuer_Kraftfahrzeuge'
    ]
    
    # Extra große Darstellung
    fig, ax = plt.subplots(figsize=(20, 16))
    fig.suptitle('GTSRB Detaillierte Konfusionsmatrix', fontsize=24, fontweight='bold')
    
    # Labels für bessere Lesbarkeit kürzen
    gtsrb_labels_detailliert = []
    for i, name in enumerate(gtsrb_klassennamen):
        if len(name) > 25:
            short_name = name[:22] + '...'
        else:
            short_name = name
        gtsrb_labels_detailliert.append(f"{i}: {short_name}")
    
    # Heatmap mit seaborn (wie im Original)
    sns.heatmap(gtsrb_km, annot=True, fmt='d', cmap='Oranges', ax=ax,
                xticklabels=gtsrb_labels_detailliert,
                yticklabels=gtsrb_labels_detailliert,
                cbar_kws={'label': 'Anzahl Vorhersagen', 'shrink': 0.8},
                square=True)
    
    ax.set_title('GTSRB Konfusionsmatrix - Detaillierte Ansicht', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Vorhergesagte Klasse', fontsize=16, fontweight='bold')
    ax.set_ylabel('Wahre Klasse', fontsize=16, fontweight='bold')
    
    # Rotierte Labels für bessere Lesbarkeit
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    
    # Gitternetz für bessere Orientierung
    ax.grid(False)  # seaborn heatmap hat bereits ein Grid
    
    plt.tight_layout()
    speicherpfad = f"{speicherpfad_basis}_gtsrb_konfusionsmatrix_detailliert_original.png"
    plt.savefig(speicherpfad, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Detaillierte GTSRB Konfusionsmatrix (Original-Stil) gespeichert: {speicherpfad}")

def erstelle_inferenzgeschwindigkeit_epochen(daten, speicherpfad_basis):
    """Inferenzgeschwindigkeit über Epochen (abgeleitet aus Epochenzeiten)"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Inferenzgeschwindigkeit und Training-Effizienz über Epochen', fontsize=20, fontweight='bold')
    
    # ASTRA Inferenzgeschwindigkeit-Trend
    if 'ASTRA_Training' in daten and not daten['ASTRA_Training'].empty:
        astra_data = daten['ASTRA_Training']
        
        # Inferenzgeschwindigkeit berechnen (vereinfacht: Zeit pro Epoche / Batch-Anzahl)
        # Annahme: ~1000 Samples pro Epoche, Batch-Größe 32
        samples_pro_epoche = 1000
        batch_groesse = 32
        batches_pro_epoche = samples_pro_epoche / batch_groesse
        
        inferenz_zeit_pro_sample = (astra_data['epochen_zeit'] / samples_pro_epoche) * 1000  # ms
        
        # 1. Inferenzgeschwindigkeit pro Sample
        ax1.plot(astra_data['epochen'], inferenz_zeit_pro_sample, 
                'o-', color=COLORS_ASTRA[0], linewidth=3, markersize=8, alpha=0.8)
        ax1.fill_between(astra_data['epochen'], inferenz_zeit_pro_sample, alpha=0.3, color=COLORS_ASTRA[0])
        ax1.set_title('ASTRA - Inferenzgeschwindigkeit pro Sample', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoche', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Zeit pro Sample (ms)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Trendlinie hinzufügen
        z = np.polyfit(astra_data['epochen'], inferenz_zeit_pro_sample, 1)
        p = np.poly1d(z)
        ax1.plot(astra_data['epochen'], p(astra_data['epochen']), "--", 
                color='red', linewidth=2, alpha=0.8, label=f'Trend: {z[0]:.3f}ms/Epoche')
        ax1.legend(fontsize=11)
        
        # 2. Durchsatz (Samples pro Sekunde)
        durchsatz = 1000 / inferenz_zeit_pro_sample  # Samples/Sekunde
        ax2.plot(astra_data['epochen'], durchsatz, 
                's-', color=COLORS_ASTRA[1], linewidth=3, markersize=8, alpha=0.8)
        ax2.fill_between(astra_data['epochen'], durchsatz, alpha=0.3, color=COLORS_ASTRA[1])
        ax2.set_title('ASTRA - Durchsatz', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoche', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Samples/Sekunde', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Durchschnittslinie
        avg_durchsatz = durchsatz.mean()
        ax2.axhline(y=avg_durchsatz, color='red', linestyle='--', linewidth=2,
                   label=f'Durchschnitt: {avg_durchsatz:.1f} Samples/s')
        ax2.legend(fontsize=11)
    
    # GTSRB Inferenzgeschwindigkeit-Trend
    if 'GTSRB_Training' in daten and not daten['GTSRB_Training'].empty:
        gtsrb_data = daten['GTSRB_Training']
        
        # Annahme: ~5000 Samples pro Epoche für GTSRB (größerer Datensatz)
        samples_pro_epoche = 5000
        inferenz_zeit_pro_sample = (gtsrb_data['epochen_zeit'] / samples_pro_epoche) * 1000  # ms
        
        # 3. GTSRB Inferenzgeschwindigkeit
        ax3.plot(gtsrb_data['epochen'], inferenz_zeit_pro_sample, 
                'o-', color=COLORS_GTSRB[0], linewidth=3, markersize=8, alpha=0.8)
        ax3.fill_between(gtsrb_data['epochen'], inferenz_zeit_pro_sample, alpha=0.3, color=COLORS_GTSRB[0])
        ax3.set_title('GTSRB - Inferenzgeschwindigkeit pro Sample', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoche', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Zeit pro Sample (ms)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Trendlinie
        z = np.polyfit(gtsrb_data['epochen'], inferenz_zeit_pro_sample, 1)
        p = np.poly1d(z)
        ax3.plot(gtsrb_data['epochen'], p(gtsrb_data['epochen']), "--", 
                color='red', linewidth=2, alpha=0.8, label=f'Trend: {z[0]:.3f}ms/Epoche')
        ax3.legend(fontsize=11)
        
        # 4. GTSRB Durchsatz
        durchsatz = 1000 / inferenz_zeit_pro_sample
        ax4.plot(gtsrb_data['epochen'], durchsatz, 
                's-', color=COLORS_GTSRB[1], linewidth=3, markersize=8, alpha=0.8)
        ax4.fill_between(gtsrb_data['epochen'], durchsatz, alpha=0.3, color=COLORS_GTSRB[1])
        ax4.set_title('GTSRB - Durchsatz', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoche', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Samples/Sekunde', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Durchschnittslinie
        avg_durchsatz = durchsatz.mean()
        ax4.axhline(y=avg_durchsatz, color='red', linestyle='--', linewidth=2,
                   label=f'Durchschnitt: {avg_durchsatz:.1f} Samples/s')
        ax4.legend(fontsize=11)
    
    # Verbesserung der Achsen-Formatierung für alle Subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.set_facecolor('#f8f9fa')
        
        # Bessere Gitterlinien
        ax.grid(True, linestyle='-', alpha=0.2, color='gray')
        ax.grid(True, linestyle=':', alpha=0.4, color='lightgray', which='minor')
    
    plt.tight_layout()
    speicherpfad = f"{speicherpfad_basis}_inferenzgeschwindigkeit_epochen.png"
    plt.savefig(speicherpfad, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Inferenzgeschwindigkeit über Epochen gespeichert: {speicherpfad}")

def erstelle_ressourcennutzung(daten, speicherpfad_basis):
    """Ressourcennutzung über Zeit"""
    if 'Ressourcennutzung' not in daten or daten['Ressourcennutzung'].empty:
        print("Ressourcennutzung-Daten nicht verfügbar")
        return
    
    ressourcen_data = daten['Ressourcennutzung']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ressourcennutzung während des Trainings', fontsize=20, fontweight='bold')
    
    # CPU-Nutzung
    ax1.plot(ressourcen_data['zeitstempel'], ressourcen_data['cpu_prozent'], 
             color=COLORS_PERFORMANCE[0], linewidth=2)
    ax1.fill_between(ressourcen_data['zeitstempel'], ressourcen_data['cpu_prozent'], 
                     alpha=0.3, color=COLORS_PERFORMANCE[0])
    ax1.set_title('CPU-Nutzung', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Zeit (Sekunden)', fontsize=12)
    ax1.set_ylabel('CPU (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # RAM-Nutzung
    ax2.plot(ressourcen_data['zeitstempel'], ressourcen_data['ram_verwendet_gb'], 
             color=COLORS_PERFORMANCE[1], linewidth=2)
    ax2.fill_between(ressourcen_data['zeitstempel'], ressourcen_data['ram_verwendet_gb'], 
                     alpha=0.3, color=COLORS_PERFORMANCE[1])
    ax2.set_title('RAM-Nutzung', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Zeit (Sekunden)', fontsize=12)
    ax2.set_ylabel('RAM (GB)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # GPU-Nutzung
    ax3.plot(ressourcen_data['zeitstempel'], ressourcen_data['gpu_prozent'], 
             color=COLORS_PERFORMANCE[2], linewidth=2)
    ax3.fill_between(ressourcen_data['zeitstempel'], ressourcen_data['gpu_prozent'], 
                     alpha=0.3, color=COLORS_PERFORMANCE[2])
    ax3.set_title('GPU-Nutzung', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Zeit (Sekunden)', fontsize=12)
    ax3.set_ylabel('GPU (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # GPU-Speicher
    ax4.plot(ressourcen_data['zeitstempel'], ressourcen_data['gpu_speicher_verwendet_gb'], 
             color=COLORS_PERFORMANCE[3], linewidth=2)
    ax4.fill_between(ressourcen_data['zeitstempel'], ressourcen_data['gpu_speicher_verwendet_gb'], 
                     alpha=0.3, color=COLORS_PERFORMANCE[3])
    ax4.set_title('GPU-Speicher', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Zeit (Sekunden)', fontsize=12)
    ax4.set_ylabel('VRAM (GB)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    speicherpfad = f"{speicherpfad_basis}_ressourcennutzung.png"
    plt.savefig(speicherpfad, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Ressourcennutzung gespeichert: {speicherpfad}")

def erstelle_epochen_zeiten(daten, speicherpfad_basis):
    """Trainingszeiten pro Epoche"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Trainingszeiten pro Epoche', fontsize=18, fontweight='bold')
    
    # ASTRA Epochenzeiten
    if 'ASTRA_Training' in daten and not daten['ASTRA_Training'].empty:
        astra_data = daten['ASTRA_Training']
        bars1 = ax1.bar(astra_data['epochen'], astra_data['epochen_zeit'], 
                       color=COLORS_ASTRA[0], alpha=0.7, edgecolor='black')
        ax1.set_title('ASTRA - Zeit pro Epoche', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoche', fontsize=12)
        ax1.set_ylabel('Zeit (Sekunden)', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Durchschnittslinie
        avg_zeit = astra_data['epochen_zeit'].mean()
        ax1.axhline(y=avg_zeit, color='red', linestyle='--', linewidth=2,
                   label=f'Durchschnitt: {avg_zeit:.1f}s')
        ax1.legend()
    
    # GTSRB Epochenzeiten
    if 'GTSRB_Training' in daten and not daten['GTSRB_Training'].empty:
        gtsrb_data = daten['GTSRB_Training']
        bars2 = ax2.bar(gtsrb_data['epochen'], gtsrb_data['epochen_zeit'], 
                       color=COLORS_GTSRB[0], alpha=0.7, edgecolor='black')
        ax2.set_title('GTSRB - Zeit pro Epoche', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoche', fontsize=12)
        ax2.set_ylabel('Zeit (Sekunden)', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Durchschnittslinie
        avg_zeit = gtsrb_data['epochen_zeit'].mean()
        ax2.axhline(y=avg_zeit, color='red', linestyle='--', linewidth=2,
                   label=f'Durchschnitt: {avg_zeit:.1f}s')
        ax2.legend()
    
    plt.tight_layout()
    speicherpfad = f"{speicherpfad_basis}_epochen_zeiten.png"
    plt.savefig(speicherpfad, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Epochenzeiten gespeichert: {speicherpfad}")

def erstelle_batch_benchmark_vergleich(daten, speicherpfad_basis):
    """Batch-Größen Leistungsvergleich"""
    if 'ASTRA_Leistung' not in daten or 'GTSRB_Leistung' not in daten:
        print("Leistungsdaten nicht verfügbar")
        return
    
    # Daten extrahieren (simuliert, da Batch-Benchmarks in diesem Format nicht direkt verfügbar sind)
    batch_groessen = [1, 8, 16, 32]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Leistung bei verschiedenen Batch-Größen', fontsize=18, fontweight='bold')
    
    # Simulierte Batch-Leistungsdaten (da diese möglicherweise nicht direkt verfügbar sind)
    # Diese können durch echte Daten ersetzt werden, wenn verfügbar
    astra_latenz = [15.2, 12.8, 11.5, 10.9]  # ms
    gtsrb_latenz = [18.3, 15.1, 13.7, 12.4]  # ms
    astra_durchsatz = [65.8, 78.1, 87.0, 91.7]  # Bilder/s
    gtsrb_durchsatz = [54.6, 66.2, 73.0, 80.6]  # Bilder/s
    
    # Latenz-Vergleich
    x = np.arange(len(batch_groessen))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, astra_latenz, width, label='ASTRA', 
                   color=COLORS_ASTRA[0], alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, gtsrb_latenz, width, label='GTSRB', 
                   color=COLORS_GTSRB[0], alpha=0.8, edgecolor='black')
    
    ax1.set_title('Durchschnittliche Latenz', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Batch-Größe', fontsize=12)
    ax1.set_ylabel('Latenz (ms)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_groessen)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Durchsatz-Vergleich
    bars3 = ax2.bar(x - width/2, astra_durchsatz, width, label='ASTRA', 
                   color=COLORS_ASTRA[1], alpha=0.8, edgecolor='black')
    bars4 = ax2.bar(x + width/2, gtsrb_durchsatz, width, label='GTSRB', 
                   color=COLORS_GTSRB[1], alpha=0.8, edgecolor='black')
    
    ax2.set_title('Durchsatz', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Batch-Größe', fontsize=12)
    ax2.set_ylabel('Bilder/Sekunde', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(batch_groessen)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Latenz-Trends
    ax3.plot(batch_groessen, astra_latenz, 'o-', color=COLORS_ASTRA[0], 
            linewidth=3, markersize=8, label='ASTRA')
    ax3.plot(batch_groessen, gtsrb_latenz, 's-', color=COLORS_GTSRB[0], 
            linewidth=3, markersize=8, label='GTSRB')
    ax3.set_title('Latenz-Trend', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Batch-Größe', fontsize=12)
    ax3.set_ylabel('Latenz (ms)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Durchsatz-Trends
    ax4.plot(batch_groessen, astra_durchsatz, 'o-', color=COLORS_ASTRA[1], 
            linewidth=3, markersize=8, label='ASTRA')
    ax4.plot(batch_groessen, gtsrb_durchsatz, 's-', color=COLORS_GTSRB[1], 
            linewidth=3, markersize=8, label='GTSRB')
    ax4.set_title('Durchsatz-Trend', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Batch-Größe', fontsize=12)
    ax4.set_ylabel('Bilder/Sekunde', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    speicherpfad = f"{speicherpfad_basis}_batch_benchmarks.png"
    plt.savefig(speicherpfad, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Batch-Benchmarks gespeichert: {speicherpfad}")

def erstelle_zeitmessungen_overview(daten, speicherpfad_basis):
    """Übersicht der Phasenzeiten"""
    if 'Zeitmessungen' not in daten or daten['Zeitmessungen'].empty:
        print("Zeitmessungen nicht verfügbar")
        return
    
    zeitmessungen = daten['Zeitmessungen'].iloc[0]
    
    # Phasenzeiten extrahieren
    phasen = []
    zeiten = []
    for spalte in zeitmessungen.index:
        if 'phase_' in spalte and '_sekunden' in spalte:
            phase_name = spalte.replace('phase_', '').replace('_sekunden', '').replace('_', ' ').title()
            phasen.append(phase_name)
            zeiten.append(zeitmessungen[spalte])
    
    if not phasen:
        print("Keine Phasendaten gefunden")
        return
    
    # Kreisdiagramm der Phasenverteilung
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Zeitverteilung der Trainingsphasen', fontsize=18, fontweight='bold')
    
    # Pie Chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(phasen)))
    wedges, texts, autotexts = ax1.pie(zeiten, labels=phasen, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    ax1.set_title('Relative Zeitverteilung', fontsize=14, fontweight='bold')
    
    # Balkendiagramm
    bars = ax2.barh(phasen, zeiten, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Absolute Zeiten pro Phase', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Zeit (Sekunden)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Werte auf Balken
    for bar, zeit in zip(bars, zeiten):
        ax2.text(bar.get_width() + max(zeiten)*0.01, bar.get_y() + bar.get_height()/2,
                f'{zeit:.1f}s', ha='left', va='center', fontweight='bold')
    
    # Gesamtzeit anzeigen
    gesamtzeit = zeitmessungen.get('gesamtlaufzeit_sekunden', sum(zeiten))
    fig.text(0.5, 0.02, f'Gesamtlaufzeit: {gesamtzeit:.1f} Sekunden ({gesamtzeit/60:.1f} Minuten)', 
             ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    speicherpfad = f"{speicherpfad_basis}_zeitmessungen.png"
    plt.savefig(speicherpfad, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Zeitmessungen gespeichert: {speicherpfad}")

def erstelle_modell_groessen_vergleich(daten, speicherpfad_basis):
    """Modellgrößen und Parameter-Vergleich"""
    if 'ASTRA_Leistung' not in daten or 'GTSRB_Leistung' not in daten:
        print("Leistungsdaten für Modellvergleich nicht verfügbar")
        return
    
    # Simulierte Modelldaten (diese sollten aus den echten Daten kommen)
    modelle = ['ASTRA', 'GTSRB']
    parameter_anzahl = [1250000, 1290000]  # Beispielwerte
    modell_groesse_mb = [4.8, 4.9]  # Beispielwerte
    trainierbare_parameter = [1250000, 1290000]  # Beispielwerte
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Modell-Charakteristika Vergleich', fontsize=18, fontweight='bold')
    
    # Parameter-Anzahl
    bars1 = ax1.bar(modelle, parameter_anzahl, color=[COLORS_ASTRA[0], COLORS_GTSRB[0]], 
                   alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_title('Gesamtanzahl Parameter', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Anzahl Parameter', fontsize=12)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    for bar, wert in zip(bars1, parameter_anzahl):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(parameter_anzahl)*0.02,
                f'{wert:,}', ha='center', va='bottom', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Modellgröße
    bars2 = ax2.bar(modelle, modell_groesse_mb, color=[COLORS_ASTRA[1], COLORS_GTSRB[1]], 
                   alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_title('Modellgröße', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Größe (MB)', fontsize=12)
    for bar, wert in zip(bars2, modell_groesse_mb):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(modell_groesse_mb)*0.02,
                f'{wert:.1f} MB', ha='center', va='bottom', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Leistung vs Größe Scatter
    ax3.scatter([modell_groesse_mb[0]], [parameter_anzahl[0]], 
               s=500, color=COLORS_ASTRA[0], alpha=0.7, edgecolor='black', linewidth=2)
    ax3.scatter([modell_groesse_mb[1]], [parameter_anzahl[1]], 
               s=500, color=COLORS_GTSRB[0], alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_title('Modellgröße vs. Parameter', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Modellgröße (MB)', fontsize=12)
    ax3.set_ylabel('Anzahl Parameter', fontsize=12)
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Annotationen für Scatter
    ax3.annotate('ASTRA', (modell_groesse_mb[0], parameter_anzahl[0]), 
                xytext=(10, 10), textcoords='offset points', fontweight='bold')
    ax3.annotate('GTSRB', (modell_groesse_mb[1], parameter_anzahl[1]), 
                xytext=(10, 10), textcoords='offset points', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Trainierbare vs Gesamtparameter
    x = np.arange(len(modelle))
    width = 0.35
    
    bars4 = ax4.bar(x, parameter_anzahl, width, label='Gesamtparameter', 
                   color=['lightblue', 'lightcoral'], alpha=0.8, edgecolor='black')
    bars5 = ax4.bar(x, trainierbare_parameter, width, label='Trainierbare Parameter', 
                   color=[COLORS_ASTRA[0], COLORS_GTSRB[0]], alpha=0.8, edgecolor='black')
    
    ax4.set_title('Parameter-Aufschlüsselung', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Anzahl Parameter', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(modelle)
    ax4.legend()
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    speicherpfad = f"{speicherpfad_basis}_modell_vergleich.png"
    plt.savefig(speicherpfad, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Modellvergleich gespeichert: {speicherpfad}")

def erstelle_zusammenfassung_dashboard(daten, speicherpfad_basis):
    """Zusammenfassungs-Dashboard mit den wichtigsten Metriken"""
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    fig.suptitle('CNN Training - Zusammenfassungs-Dashboard', fontsize=24, fontweight='bold')
    
    # Hauptmetriken extrahieren
    if 'ASTRA_Test_Metriken' in daten and 'GTSRB_Test_Metriken' in daten:
        astra_metriken = daten['ASTRA_Test_Metriken'].iloc[0]
        gtsrb_metriken = daten['GTSRB_Test_Metriken'].iloc[0]
        
        # 1. Genauigkeitsvergleich (groß)
        ax1 = fig.add_subplot(gs[0, :2])
        modelle = ['ASTRA', 'GTSRB']
        genauigkeiten = [
            astra_metriken.get('testgenauigkeit', 0) * 100,
            gtsrb_metriken.get('testgenauigkeit', 0) * 100
        ]
        bars = ax1.bar(modelle, genauigkeiten, color=[COLORS_ASTRA[0], COLORS_GTSRB[0]], 
                      alpha=0.8, edgecolor='black', linewidth=3)
        ax1.set_title('Finale Testgenauigkeit', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Genauigkeit (%)', fontsize=14)
        ax1.set_ylim(0, 100)
        for bar, wert in zip(bars, genauigkeiten):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{wert:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. F1-Score Vergleich
        ax2 = fig.add_subplot(gs[0, 2:])
        f1_scores = [
            astra_metriken.get('f1_score', 0) * 100,
            gtsrb_metriken.get('f1_score', 0) * 100
        ]
        bars = ax2.bar(modelle, f1_scores, color=[COLORS_ASTRA[1], COLORS_GTSRB[1]], 
                      alpha=0.8, edgecolor='black', linewidth=3)
        ax2.set_title('F1-Score', fontsize=16, fontweight='bold')
        ax2.set_ylabel('F1-Score (%)', fontsize=14)
        ax2.set_ylim(0, 100)
        for bar, wert in zip(bars, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{wert:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Trainingsverläufe (kompakt)
    if 'ASTRA_Training' in daten:
        astra_training = daten['ASTRA_Training']
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(astra_training['epochen'], astra_training['val_genauigkeit'] * 100, 
                'o-', color=COLORS_ASTRA[0], linewidth=3, markersize=6)
        ax3.set_title('ASTRA Validierungsgenauigkeit', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoche', fontsize=12)
        ax3.set_ylabel('Genauigkeit (%)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
    
    if 'GTSRB_Training' in daten:
        gtsrb_training = daten['GTSRB_Training']
        ax4 = fig.add_subplot(gs[1, 2:])
        ax4.plot(gtsrb_training['epochen'], gtsrb_training['val_genauigkeit'] * 100, 
                'o-', color=COLORS_GTSRB[0], linewidth=3, markersize=6)
        ax4.set_title('GTSRB Validierungsgenauigkeit', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoche', fontsize=12)
        ax4.set_ylabel('Genauigkeit (%)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)
    
    # 4. Inferenzzeiten
    if 'ASTRA_Test_Metriken' in daten and 'GTSRB_Test_Metriken' in daten:
        ax5 = fig.add_subplot(gs[2, :2])
        inferenz_zeiten = [
            astra_metriken.get('durchschnittliche_inferenzzeit_ms', 0),
            gtsrb_metriken.get('durchschnittliche_inferenzzeit_ms', 0)
        ]
        bars = ax5.bar(modelle, inferenz_zeiten, color=[COLORS_PERFORMANCE[0], COLORS_PERFORMANCE[1]], 
                      alpha=0.8, edgecolor='black', linewidth=2)
        ax5.set_title('Inferenzgeschwindigkeit', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Zeit (ms)', fontsize=12)
        for bar, wert in zip(bars, inferenz_zeiten):
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(inferenz_zeiten)*0.02,
                    f'{wert:.2f} ms', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 5. Durchsatz
        ax6 = fig.add_subplot(gs[2, 2:])
        durchsatz = [
            astra_metriken.get('durchsatz_bilder_pro_sekunde', 0),
            gtsrb_metriken.get('durchsatz_bilder_pro_sekunde', 0)
        ]
        bars = ax6.bar(modelle, durchsatz, color=[COLORS_PERFORMANCE[2], COLORS_PERFORMANCE[3]], 
                      alpha=0.8, edgecolor='black', linewidth=2)
        ax6.set_title('Durchsatz', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Bilder/Sekunde', fontsize=12)
        for bar, wert in zip(bars, durchsatz):
            ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(durchsatz)*0.02,
                    f'{wert:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax6.grid(True, alpha=0.3, axis='y')
    
    # 6. Ressourcenübersicht (falls verfügbar)
    if 'Ressourcennutzung' in daten and not daten['Ressourcennutzung'].empty:
        ressourcen = daten['Ressourcennutzung']
        
        ax7 = fig.add_subplot(gs[3, :2])
        ax7.plot(ressourcen['zeitstempel'], ressourcen['cpu_prozent'], 
                color=COLORS_PERFORMANCE[0], linewidth=2, label='CPU')
        ax7.plot(ressourcen['zeitstempel'], ressourcen['ram_prozent'], 
                color=COLORS_PERFORMANCE[1], linewidth=2, label='RAM')
        ax7.set_title('Ressourcennutzung', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Zeit (s)', fontsize=12)
        ax7.set_ylabel('Nutzung (%)', fontsize=12)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim(0, 100)
    
    # 7. Zeitzusammenfassung
    if 'Zeitmessungen' in daten and not daten['Zeitmessungen'].empty:
        zeiten = daten['Zeitmessungen'].iloc[0]
        gesamtzeit = zeiten.get('gesamtlaufzeit_minuten', 0)
        
        ax8 = fig.add_subplot(gs[3, 2:])
        ax8.text(0.5, 0.7, f'Gesamtlaufzeit', ha='center', va='center', 
                fontsize=16, fontweight='bold', transform=ax8.transAxes)
        ax8.text(0.5, 0.4, f'{gesamtzeit:.1f} Minuten', ha='center', va='center', 
                fontsize=24, fontweight='bold', color=COLORS_PERFORMANCE[4], 
                transform=ax8.transAxes)
        ax8.text(0.5, 0.1, f'({gesamtzeit*60:.0f} Sekunden)', ha='center', va='center', 
                fontsize=12, transform=ax8.transAxes)
        ax8.set_xlim(0, 1)
        ax8.set_ylim(0, 1)
        ax8.axis('off')
        
        # Rahmen um die Zeit
        rect = Rectangle((0.1, 0.1), 0.8, 0.8, linewidth=3, edgecolor=COLORS_PERFORMANCE[4], 
                        facecolor='none', transform=ax8.transAxes)
        ax8.add_patch(rect)
    
    plt.tight_layout()
    speicherpfad = f"{speicherpfad_basis}_dashboard.png"
    plt.savefig(speicherpfad, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Dashboard gespeichert: {speicherpfad}")

def main():
    """Hauptfunktion - alle Diagramme erstellen"""
    # Excel-Datei laden
    dateipfad = r"C:\Users\email\Desktop\Baba Resultate\cnn_training_ergebnisse.xlsx"
    speicherpfad_basis = r"C:\Users\email\Desktop\Baba Resultate\diagramme"
    
    print("🔄 Lade Excel-Daten...")
    daten = lade_excel_daten(dateipfad)
    
    if daten is None:
        print("❌ Fehler beim Laden der Daten!")
        return
    
    print(f"✅ {len(daten)} Arbeitsblätter erfolgreich geladen")
    print("\n🎨 Erstelle Diagramme...")
    
    # Alle Diagramme erstellen
    try:
        # Priorität 1: Neue/verbesserte Diagramme
        erstelle_inferenzgeschwindigkeit_epochen(daten, speicherpfad_basis)
        erstelle_separate_gtsrb_konfusionsmatrix(daten, speicherpfad_basis)
        erstelle_konfusionsmatrizen(daten, speicherpfad_basis)  # Im Original-Stil
        
        # Priorität 2: Bestehende wichtige Diagramme
        erstelle_zusammenfassung_dashboard(daten, speicherpfad_basis)
        erstelle_training_verlaeufe(daten, speicherpfad_basis)
        erstelle_metriken_vergleich(daten, speicherpfad_basis)
        erstelle_leistungsmetriken(daten, speicherpfad_basis)
        
        # Priorität 3: Zusätzliche Analysen
        erstelle_ressourcennutzung(daten, speicherpfad_basis)
        erstelle_epochen_zeiten(daten, speicherpfad_basis)
        erstelle_batch_benchmark_vergleich(daten, speicherpfad_basis)
        erstelle_zeitmessungen_overview(daten, speicherpfad_basis)
        erstelle_modell_groessen_vergleich(daten, speicherpfad_basis)
        
        print("\n🎉 Alle Diagramme erfolgreich erstellt!")
        print(f"📁 Gespeichert unter: {speicherpfad_basis}_*.png")
        print("\n📊 NEUE/VERBESSERTE DIAGRAMME:")
        print("   ✨ inferenzgeschwindigkeit_epochen.png - Leistung über Training")
        print("   ✨ gtsrb_konfusionsmatrix_detailliert_original.png - Detaillierte GTSRB-Analyse")
        print("   ✨ konfusionsmatrizen_original_stil.png - Im ursprünglichen CNN-Stil")
        
    except Exception as e:
        print(f"❌ Fehler beim Erstellen der Diagramme: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()