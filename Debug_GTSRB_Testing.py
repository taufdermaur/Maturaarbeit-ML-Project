import torch
import torch.nn as nn
import os
from torchvision import transforms, datasets
from PIL import Image, ImageOps
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
import re



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_gtsrb_model():
    try:
        from torchvision import transforms
        
        class TrafficSignCNN(nn.Module):
            def __init__(self, num_classes=43):
                super(TrafficSignCNN, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5), nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True),
                    nn.Dropout(0.5), nn.Linear(512, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                return self.classifier(x.view(x.size(0), -1))
        
        gtsrb_transform = transforms.Compose([
            transforms.Resize((64, 64)), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        gtsrb_model_path = r'C:\Users\timau\Desktop\gtsrb_model.pth'
        if os.path.exists(gtsrb_model_path):
            gtsrb_model = TrafficSignCNN(43).to(device)
            gtsrb_model.load_state_dict(torch.load(gtsrb_model_path, map_location=device))
            gtsrb_model.eval()
            print("GTSRB classification model loaded successfully")
            return gtsrb_model, gtsrb_transform
        else:
            print(f"GTSRB model not found at {gtsrb_model_path}")
            return None, None
    except Exception as e:
        print(f"Error loading GTSRB model: {e}")
        return None, None

gtsrb_model, gtsrb_transform = load_gtsrb_model()

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Custom Dataset class for GTSRB test data (uses CSV)
class GTSRBTestDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None, class_to_idx=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['Path'])
        image = Image.open(img_path).convert('RGB')
        class_id = self.annotations.iloc[idx]['ClassId']
        
        # Map the actual class ID to the training dataset's index
        if self.class_to_idx is not None:
            folder_name = f"{class_id}_"
            mapped_class = None
            for folder, idx in self.class_to_idx.items():
                if folder.startswith(folder_name):
                    mapped_class = idx
                    break
            
            if mapped_class is not None:
                class_id = mapped_class
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_id

print("Preparing GTSRB test dataset...")

# Load GTSRB dataset
gtsrb_train_dataset = datasets.ImageFolder(
    root=r'C:\Users\timau\Desktop\Datensaetze\GTSRB\Train',
    transform=train_transform
)

gtsrb_test_dataset = GTSRBTestDataset(
    csv_file=r'C:\Users\timau\Desktop\Datensaetze\GTSRB\Test.csv',
    root_dir=r'C:\Users\timau\Desktop\Datensaetze\GTSRB',
    transform=test_transform,
    class_to_idx=gtsrb_train_dataset.class_to_idx
)

gtsrb_test_loader = DataLoader(gtsrb_test_dataset, batch_size=32, shuffle=False)

# GTSRB Model Evaluation
print("\nEvaluating GTSRB model...")
gtsrb_model.eval()
gtsrb_all_predictions = []
gtsrb_all_targets = []
gtsrb_all_confidences = []
gtsrb_class_confidences = {i: [] for i in range(len(gtsrb_train_dataset.classes))}

with torch.no_grad():
    for data, target in gtsrb_test_loader:
        data, target = data.to(device), target.to(device)
        output = gtsrb_model(data)
        probabilities = torch.softmax(output, dim=1)
        confidences, predicted = torch.max(probabilities, 1)
        
        gtsrb_all_predictions.extend(predicted.cpu().numpy())
        gtsrb_all_targets.extend(target.cpu().numpy())
        gtsrb_all_confidences.extend(confidences.cpu().numpy())
        
        # Collect confidences per class
        for i, (pred, conf, true_label) in enumerate(zip(predicted.cpu().numpy(), 
                                                       confidences.cpu().numpy(), 
                                                       target.cpu().numpy())):
            if pred == true_label:  # Only correct predictions
                gtsrb_class_confidences[true_label].append(conf)

gtsrb_accuracy = accuracy_score(gtsrb_all_targets, gtsrb_all_predictions)
gtsrb_precision, gtsrb_recall, gtsrb_f1, _ = precision_recall_fscore_support(gtsrb_all_targets, gtsrb_all_predictions, average='weighted')

print(f"GTSRB Test Accuracy: {gtsrb_accuracy*100:.2f}%")
print(f"GTSRB Test Precision: {gtsrb_precision*100:.2f}%")
print(f"GTSRB Test Recall: {gtsrb_recall*100:.2f}%")
print(f"GTSRB Test F1-Score: {gtsrb_f1*100:.2f}%")


test_data = r"G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results\debug_patches"

def extract_label(filename):
    match = re.search(r'class(\d+)', filename)
    return int(match.group(1)) if match else None

def pad_patch(img, pad_px=16):
    return ImageOps.expand(img, border=pad_px, fill=(128,128,128))  # Grau als Hintergrund

def test_gtsrb_model_on_folder(model, transform, folder):
    correct = 0
    total = 0
    for fname in os.listdir(folder):
        if fname.endswith('.png'):
            label = extract_label(fname)
            if label is None:
                continue
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert('RGB')
            img = pad_patch(img, pad_px=16)  # <-- Padding hinzufügen
            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred = output.argmax(dim=1).item()
                if pred == label:
                    correct += 1
                total += 1
    print(f"Tested {total} images. Accuracy: {100 * correct / total:.2f}%")

if gtsrb_model is not None:
    test_gtsrb_model_on_folder(gtsrb_model, gtsrb_transform, test_data)

def test_gtsrb_model_on_folder(model, transform, folder):
    correct = 0
    total = 0
    for idx, fname in enumerate(os.listdir(folder)):
        if fname.endswith('.jpg') or fname.endswith('.png'):
            label = extract_label(fname)
            if label is None:
                continue
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert('RGB')
            img = pad_patch(img, pad_px=16)  # <-- Padding hinzufügen
            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred = output.argmax(dim=1).item()
                if idx < 10:  # Debug-Ausgabe für die ersten 10 Bilder
                    print(f"File: {fname}, Label: {label}, Prediction: {pred}")
                if pred == label:
                    correct += 1
                total += 1
    print(f"Tested {total} images. Accuracy: {100 * correct / total:.2f}%")

test_gtsrb_model_on_folder(gtsrb_model, gtsrb_transform, test_data)


test_gtsrb_model_on_folder(gtsrb_model, gtsrb_transform, test_data)
