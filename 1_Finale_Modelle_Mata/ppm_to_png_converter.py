from PIL import Image
import os

def ppm_to_png_converter(folder_path):
    """
    Konvertiert alle .ppm-Dateien in einem angegebenen Ordner in .png-Dateien.

    Args:
        folder_path (str): Der Pfad zu dem Ordner, der die .ppm-Dateien enthält.
    """
    # Überprüfen, ob der angegebene Pfad ein Ordner ist
    if not os.path.isdir(folder_path):
        print(f"Fehler: Der Pfad '{folder_path}' ist kein gültiger Ordner.")
        return

    # Durchlaufen aller Dateien im Ordner
    for filename in os.listdir(folder_path):
        if filename.endswith(".ppm"):
            ppm_file_path = os.path.join(folder_path, filename)
            # Dateinamen ohne Endung für die neue .png-Datei
            file_name_without_extension = os.path.splitext(filename)[0]
            png_file_path = os.path.join(folder_path, f"{file_name_without_extension}.png")

            try:
                # Öffnen der .ppm-Datei
                with Image.open(ppm_file_path) as img:
                    # Speichern als .png-Datei
                    img.save(png_file_path, "PNG")
                print(f"Konvertiert: '{filename}' -> '{file_name_without_extension}.png'")
            except Exception as e:
                print(f"Fehler bei der Konvertierung von '{filename}': {e}")

# Beispielaufruf des Skripts
if __name__ == "__main__":
    # Ersetze 'dein_ordnerpfad' durch den tatsächlichen Pfad zu deinem Ordner.
    # Beispiel:
    # ordner_mit_ppm_dateien = "/pfad/zu/deinem/ordner"
    ordner_mit_ppm_dateien = r"C:\Users\timau\Desktop\Datensaetze\GTSDB\Test"
    ppm_to_png_converter(ordner_mit_ppm_dateien)