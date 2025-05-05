import os
import zipfile
import shutil
import json
import uuid
from PIL import Image
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import subprocess
import glob
import tempfile

class TrafficSignLabeler:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = None
        self.labels = {}
        self.categories = []
        
    def select_zip_file(self):
        """Prompt user to select the ZIP file containing traffic signs."""
        zip_path = filedialog.askopenfilename(
            title="Wähle das ZIP-Archiv mit den Verkehrsschildern",
            filetypes=[("ZIP files", "*.zip")]
        )
        if not zip_path:
            print("Keine ZIP-Datei ausgewählt. Programm wird beendet.")
            exit()
        return zip_path
    
    def select_output_directory(self):
        """Prompt user to select output directory for labeled dataset."""
        output_dir = filedialog.askdirectory(
            title="Wähle das Ausgabeverzeichnis für den gelabelten Datensatz"
        )
        if not output_dir:
            print("Kein Ausgabeverzeichnis ausgewählt. Programm wird beendet.")
            exit()
        
        # Create required directories
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
        return output_dir
    
    def extract_zip(self, zip_path):
        """Extract the ZIP file to a temporary directory."""
        print(f"Extrahiere {zip_path} nach {self.temp_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)
    
    def convert_to_jpeg(self, file_path):
        """Convert EPS or WMF file to JPEG using appropriate tools."""
        file_ext = os.path.splitext(file_path)[1].lower()
        output_path = os.path.splitext(file_path)[0] + ".jpg"
        
        if file_ext == '.eps':
            # Using Ghostscript for EPS conversion
            try:
                # Check if ghostscript is installed
                subprocess.run(["gs", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Convert EPS to JPEG
                subprocess.run([
                    "gs", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=jpeg", 
                    "-r300", f"-sOutputFile={output_path}", file_path
                ])
                return output_path
            except FileNotFoundError:
                print("Ghostscript nicht gefunden. Bitte installiere Ghostscript für EPS-Konvertierung.")
                return None
        
        elif file_ext == '.wmf':
            # Using ImageMagick for WMF conversion
            try:
                # Check if ImageMagick is installed
                subprocess.run(["convert", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Convert WMF to JPEG
                subprocess.run([
                    "convert", file_path, "-quality", "90", output_path
                ])
                return output_path
            except FileNotFoundError:
                print("ImageMagick nicht gefunden. Bitte installiere ImageMagick für WMF-Konvertierung.")
                return None
        
        else:
            print(f"Unbekanntes Dateiformat: {file_ext}")
            return None
    
    def create_category(self, name):
        """Add a new category to the dataset."""
        category_id = len(self.categories) + 1
        category = {
            "id": category_id,
            "name": name,
            "supercategory": "traffic_sign"
        }
        self.categories.append(category)
        return category_id
    
    def get_category_from_path(self, file_path):
        """Extract category information from file path or prompt user."""
        # Get directory name as a potential category
        dir_name = os.path.basename(os.path.dirname(file_path))
        
        # Present dialog to confirm/modify category
        category_name = simpledialog.askstring(
            "Kategorie bestätigen",
            f"Kategorie für {os.path.basename(file_path)}:",
            initialvalue=dir_name
        )
        
        # Find existing category or create new one
        for cat in self.categories:
            if cat["name"] == category_name:
                return cat["id"]
        
        # Create new category if it doesn't exist
        return self.create_category(category_name)
    
    def process_files(self):
        """Process all files in the extracted directory."""
        # Find all EPS and WMF files
        eps_files = glob.glob(os.path.join(self.temp_dir, "**", "*.eps"), recursive=True)
        wmf_files = glob.glob(os.path.join(self.temp_dir, "**", "*.wmf"), recursive=True)
        all_files = eps_files + wmf_files
        
        print(f"Gefunden: {len(eps_files)} EPS-Dateien und {len(wmf_files)} WMF-Dateien")
        
        # Create COCO format annotations structure
        annotations = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        annotation_id = 1
        
        # Process each file
        for file_idx, file_path in enumerate(all_files):
            print(f"Verarbeite {file_idx+1}/{len(all_files)}: {file_path}")
            
            # Convert file to JPEG
            jpeg_path = self.convert_to_jpeg(file_path)
            if not jpeg_path:
                print(f"Konnte {file_path} nicht konvertieren. Überspringe...")
                continue
            
            try:
                # Open the image to get dimensions
                with Image.open(jpeg_path) as img:
                    width, height = img.size
                
                # Generate unique image ID and filename
                image_id = file_idx + 1
                new_filename = f"sign_{image_id:06d}.jpg"
                
                # Copy to output directory
                output_image_path = os.path.join(self.output_dir, "images", new_filename)
                shutil.copy2(jpeg_path, output_image_path)
                
                # Get category
                category_id = self.get_category_from_path(file_path)
                
                # Add image to COCO structure
                annotations["images"].append({
                    "id": image_id,
                    "file_name": new_filename,
                    "width": width,
                    "height": height
                })
                
                # Add annotation to COCO structure
                annotations["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [0, 0, width, height],  # Full image bounding box
                    "area": width * height,
                    "iscrowd": 0
                })
                
                annotation_id += 1
                
            except Exception as e:
                print(f"Fehler bei der Verarbeitung von {file_path}: {e}")
        
        # Add categories to annotations
        annotations["categories"] = self.categories
        
        # Save annotations to JSON file
        with open(os.path.join(self.output_dir, "annotations", "instances.json"), "w") as f:
            json.dump(annotations, f, indent=2)
        
        print(f"Dataset erstellt im Verzeichnis: {self.output_dir}")
        print(f"Anzahl der verarbeiteten Bilder: {len(annotations['images'])}")
        print(f"Anzahl der erstellten Kategorien: {len(annotations['categories'])}")
        
        # Show completion message
        messagebox.showinfo(
            "Verarbeitung abgeschlossen",
            f"Dataset mit {len(annotations['images'])} Bildern und {len(annotations['categories'])} Kategorien wurde erstellt."
        )
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
            print(f"Temporäre Dateien entfernt: {self.temp_dir}")
        except Exception as e:
            print(f"Fehler beim Entfernen temporärer Dateien: {e}")
    
    def run(self):
        """Main execution flow."""
        print("Schweizer Verkehrsschilder Labeling Tool")
        print("=======================================")
        
        try:
            # Get input and output paths
            zip_path = self.select_zip_file()
            self.select_output_directory()
            
            # Process files
            self.extract_zip(zip_path)
            self.process_files()
            
            # Split dataset into train/val/test
            self.create_dataset_splits()
            
        finally:
            # Clean up
            self.cleanup()
            
    def create_dataset_splits(self):
        """Create train/val/test splits from the dataset."""
        import random
        
        # Load annotations
        annotations_path = os.path.join(self.output_dir, "annotations", "instances.json")
        with open(annotations_path, "r") as f:
            data = json.load(f)
        
        # Get list of image IDs
        image_ids = [img["id"] for img in data["images"]]
        random.shuffle(image_ids)
        
        # Split into train (70%), validation (15%), test (15%)
        train_size = int(len(image_ids) * 0.7)
        val_size = int(len(image_ids) * 0.15)
        
        train_ids = set(image_ids[:train_size])
        val_ids = set(image_ids[train_size:train_size + val_size])
        test_ids = set(image_ids[train_size + val_size:])
        
        # Create split data
        splits = {
            "train": {"images": [], "annotations": []},
            "val": {"images": [], "annotations": []},
            "test": {"images": [], "annotations": []}
        }
        
        # Split images
        for img in data["images"]:
            if img["id"] in train_ids:
                splits["train"]["images"].append(img)
            elif img["id"] in val_ids:
                splits["val"]["images"].append(img)
            else:
                splits["test"]["images"].append(img)
        
        # Split annotations
        for ann in data["annotations"]:
            if ann["image_id"] in train_ids:
                splits["train"]["annotations"].append(ann)
            elif ann["image_id"] in val_ids:
                splits["val"]["annotations"].append(ann)
            else:
                splits["test"]["annotations"].append(ann)
        
        # Add categories to all splits
        for split_name in splits:
            splits[split_name]["categories"] = data["categories"]
        
        # Save split files
        for split_name, split_data in splits.items():
            output_path = os.path.join(self.output_dir, "annotations", f"instances_{split_name}.json")
            with open(output_path, "w") as f:
                json.dump(split_data, f, indent=2)
        
        # Print statistics
        print("\nDataset-Splits erstellt:")
        print(f"Training: {len(splits['train']['images'])} Bilder")
        print(f"Validierung: {len(splits['val']['images'])} Bilder")
        print(f"Test: {len(splits['test']['images'])} Bilder")


if __name__ == "__main__":
    labeler = TrafficSignLabeler()
    labeler.run()