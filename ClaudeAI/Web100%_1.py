import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import os
import base64
import io
from flask import Flask, render_template_string, request, jsonify
import threading
import webbrowser
from datetime import datetime

# Set random seed for reproducibility
torch.manual_seed(42)

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Custom Dataset Class
class ImageFolderDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = Path(root_folder)
        self.transform = transform
        
        # Get classes from folder names
        self.classes = sorted([d.name for d in self.root_folder.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.samples = []
        for cls_name in self.classes:
            cls_folder = self.root_folder / cls_name
            for img_path in cls_folder.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    self.samples.append((img_path, self.class_to_idx[cls_name]))
        
        print(f"Found {len(self.samples)} images in {len(self.classes)} classes")
        print(f"Classes: {self.class_to_idx}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load image using PIL
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                black_img = Image.new('RGB', (128, 128), (0, 0, 0))
                return self.transform(black_img), label
            return None, label

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
    transforms.RandomRotation(10),           # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# Dataset setup
train_folder = r"G:\Meine Ablage\Maturaarbeit\Programmierung_Python\First-Steps-ML-Project\dataset\train"
test_folder = r"G:\Meine Ablage\Maturaarbeit\Programmierung_Python\First-Steps-ML-Project\dataset\test"

# Create test transform (no augmentation for testing)
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = ImageFolderDataset(train_folder, transform=train_transform)
test_dataset = ImageFolderDataset(test_folder, transform=test_transform)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Improved CNN Model
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize model, loss, and optimizer
num_classes = len(train_dataset.classes)
model = ImprovedCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# Test function
def test_model(model, test_loader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}% ({correct}/{total})")
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"{class_name}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"{class_name}: No samples")
    
    return test_loss, test_acc

# Training loop
epochs = 200
train_losses, train_accs = [], []

print("Starting training...")
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print("-" * 30)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
ax1.plot(range(1, epochs+1), train_losses, 'b-', label='Training Loss')
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Accuracy plot
ax2.plot(range(1, epochs+1), train_accs, 'b-', label='Training Accuracy')
ax2.set_title('Training Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Save the model
torch.save(model.state_dict(), 'improved_cnn_model.pth')
print("Model saved as 'improved_cnn_model.pth'")

# Test the model on test dataset
print("\n" + "="*50)
print("TESTING PHASE")
print("="*50)
test_loss, test_acc = test_model(model, test_loader, criterion, device, train_dataset.classes)

# Create prediction function for web interface
def predict_image(image, model, transform, class_names, device):
    """Predict class of a single image"""
    model.eval()
    
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        
    return class_names[predicted.item()], confidence.item() * 100

# Enhanced Web interface HTML template with file upload
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Classification - Camera & Upload</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            text-align: center;
        }
        h1 {
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .method-tabs {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 5px;
            backdrop-filter: blur(10px);
        }
        .tab {
            flex: 1;
            padding: 15px 20px;
            background: transparent;
            border: none;
            color: white;
            cursor: pointer;
            border-radius: 10px;
            transition: all 0.3s ease;
            font-size: 16px;
        }
        .tab.active {
            background: rgba(255,255,255,0.2);
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .tab:hover {
            background: rgba(255,255,255,0.15);
        }
        .method-content {
            display: none;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .method-content.active {
            display: block;
        }
        
        /* File Upload Styles */
        .upload-area {
            border: 3px dashed rgba(255,255,255,0.3);
            border-radius: 15px;
            padding: 40px 20px;
            margin: 20px 0;
            transition: all 0.3s ease;
            cursor: pointer;
            background: rgba(255,255,255,0.05);
        }
        .upload-area:hover, .upload-area.dragover {
            border-color: rgba(255,255,255,0.6);
            background: rgba(255,255,255,0.1);
            transform: translateY(-2px);
        }
        .upload-icon {
            font-size: 3em;
            margin-bottom: 15px;
            opacity: 0.7;
        }
        .upload-text {
            font-size: 1.2em;
            margin-bottom: 10px;
        }
        .upload-subtext {
            font-size: 0.9em;
            opacity: 0.7;
        }
        #fileInput {
            display: none;
        }
        .preview-container {
            margin: 20px 0;
            display: none;
        }
        .preview-image {
            max-width: 100%;
            max-height: 300px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        
        /* Camera Styles */
        .camera-container {
            position: relative;
        }
        video, canvas {
            width: 100%;
            max-width: 400px;
            height: 300px;
            object-fit: cover;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        #canvas {
            display: none;
        }
        
        /* Common Styles */
        .controls {
            margin: 20px 0;
        }
        button {
            background: rgba(255,255,255,0.2);
            border: 2px solid rgba(255,255,255,0.3);
            color: white;
            padding: 12px 24px;
            margin: 5px;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        button:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }
        button:active {
            transform: translateY(0);
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .prediction {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 10px;
            color: #FFD700;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }
        .confidence {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .status {
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
            font-weight: bold;
        }
        .loading {
            background: rgba(255,193,7,0.2);
            color: #FFC107;
        }
        .error {
            background: rgba(220,53,69,0.2);
            color: #DC3545;
        }
        .success {
            background: rgba(40,167,69,0.2);
            color: #28A745;
        }
        .info {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            font-size: 0.9em;
            backdrop-filter: blur(10px);
        }
        .classes-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
            margin: 15px 0;
        }
        .class-tag {
            background: rgba(255,255,255,0.2);
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚦 Sign Classification</h1>
        
        <div class="method-tabs">
            <button class="tab active" onclick="switchMethod('upload')">📱 Upload Photo</button>
            <button class="tab" onclick="switchMethod('camera')">📷 Live Camera</button>
        </div>
        
        <!-- Upload Method -->
        <div id="upload-method" class="method-content active">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()" 
                 ondrop="handleDrop(event)" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)">
                <div class="upload-icon">📸</div>
                <div class="upload-text">Take a photo with your camera app</div>
                <div class="upload-subtext">Then tap here to upload • Or drag & drop</div>
                <div class="upload-subtext" style="margin-top: 10px; font-size: 0.8em;">
                    Supports: JPG, PNG, HEIC, WebP
                </div>
            </div>
            <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
            
            <div id="preview-container" class="preview-container">
                <img id="preview-image" class="preview-image" alt="Preview">
                <div class="controls">
                    <button onclick="classifyUploadedImage()">🔍 Classify Image</button>
                    <button onclick="clearUpload()">🗑️ Clear</button>
                </div>
            </div>
        </div>
        
        <!-- Camera Method -->
        <div id="camera-method" class="method-content">
            <div class="camera-container">
                <video id="video" playsinline></video>
                <canvas id="canvas"></canvas>
            </div>
            
            <div class="controls">
                <button onclick="startCamera()">📷 Start Camera</button>
                <button onclick="captureImage()">📸 Capture & Predict</button>
                <button onclick="stopCamera()">⏹️ Stop Camera</button>
            </div>
        </div>
        
        <div id="status" class="status" style="display:none;"></div>
        
        <div id="result" class="result" style="display:none;">
            <div id="prediction" class="prediction"></div>
            <div id="confidence" class="confidence"></div>
            <div id="timestamp"></div>
        </div>
        
        <div class="info">
            <strong>📋 Detectable Classes:</strong>
            <div class="classes-list">
                {% for class_name in classes %}
                <span class="class-tag">{{ class_name }}</span>
                {% endfor %}
            </div>
            
            <br><strong>📱 Recommended for iPhone:</strong><br>
            1. Use the <strong>"Upload Photo"</strong> method<br>
            2. Take photos with your Camera app<br>
            3. Upload them here for classification<br>
            <br>
            <strong>🔧 If using Live Camera:</strong><br>
            • Use Safari browser only<br>
            • Allow camera permissions<br>
            • Works best on computers/laptops<br>
        </div>
    </div>

    <script>
        let video = document.getElementById('video');
        let canvas = document.getElementById('canvas');
        let ctx = canvas.getContext('2d');
        let stream = null;
        let uploadedImageData = null;
        
        function switchMethod(method) {
            // Update tabs
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.method-content').forEach(content => content.classList.remove('active'));
            
            if (method === 'upload') {
                document.querySelector('.tab:first-child').classList.add('active');
                document.getElementById('upload-method').classList.add('active');
            } else {
                document.querySelector('.tab:last-child').classList.add('active');  
                document.getElementById('camera-method').classList.add('active');
            }
            
            // Clear any previous results
            clearResult();
        }
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + type;
            status.style.display = 'block';
            
            if (type !== 'loading') {
                setTimeout(() => {
                    status.style.display = 'none';
                }, 5000);
            }
        }
        
        function clearResult() {
            document.getElementById('result').style.display = 'none';
            document.getElementById('status').style.display = 'none';
        }
        
        // File Upload Functions
        function handleDragOver(e) {
            e.preventDefault();
            e.currentTarget.classList.add('dragover');
        }
        
        function handleDragLeave(e) {
            e.currentTarget.classList.remove('dragover');
        }
        
        function handleDrop(e) {
            e.preventDefault();
            e.currentTarget.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                processFile(files[0]);
            }
        }
        
        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (file) {
                processFile(file);
            }
        }
        
        function processFile(file) {
            // Check file type
            const validTypes = ['image/jpeg', 'image/png', 'image/heic', 'image/webp', 'image/bmp'];
            if (!validTypes.includes(file.type) && !file.name.toLowerCase().match(/\.(jpg|jpeg|png|heic|webp|bmp)$/)) {
                showStatus('Please select a valid image file (JPG, PNG, HEIC, WebP, BMP)', 'error');
                return;
            }
            
            // Check file size (limit to 10MB)
            if (file.size > 10 * 1024 * 1024) {
                showStatus('File too large. Please select an image under 10MB.', 'error');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.getElementById('preview-image');
                img.src = e.target.result;
                document.getElementById('preview-container').style.display = 'block';
                uploadedImageData = e.target.result;
                clearResult();
            };
            reader.readAsDataURL(file);
        }
        
        function clearUpload() {
            document.getElementById('preview-container').style.display = 'none';
            document.getElementById('fileInput').value = '';
            uploadedImageData = null;
            clearResult();
        }
        
        async function classifyUploadedImage() {
            if (!uploadedImageData) {
                showStatus('Please select an image first!', 'error');
                return;
            }
            
            try {
                showStatus('Analyzing image...', 'loading');
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: uploadedImageData
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('prediction').textContent = result.prediction;
                    document.getElementById('confidence').textContent = 
                        `Confidence: ${result.confidence.toFixed(1)}%`;
                    document.getElementById('timestamp').textContent = 
                        `Analyzed at: ${new Date().toLocaleTimeString()}`;
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('status').style.display = 'none';
                } else {
                    showStatus('Classification failed: ' + result.error, 'error');
                }
                
            } catch (error) {
                console.error('Error:', error);
                showStatus('Error processing image. Please try again.', 'error');
            }
        }
        
        // Camera Functions (existing code)
        async function startCamera() {
            try {
                showStatus('Starting camera...', 'loading');
                
                const configurations = [
                    {
                        video: {
                            facingMode: { exact: 'environment' },
                            width: { ideal: 640 },
                            height: { ideal: 480 }
                        }
                    },
                    {
                        video: {
                            facingMode: 'environment',
                            width: { ideal: 640 },
                            height: { ideal: 480 }
                        }
                    },
                    {
                        video: {
                            facingMode: 'user',
                            width: { ideal: 640 },
                            height: { ideal: 480 }
                        }
                    },
                    {
                        video: {
                            width: { ideal: 640 },
                            height: { ideal: 480 }
                        }
                    },
                    {
                        video: true
                    }
                ];
                
                let success = false;
                for (let config of configurations) {
                    try {
                        console.log('Trying camera config:', config);
                        stream = await navigator.mediaDevices.getUserMedia(config);
                        success = true;
                        break;
                    } catch (e) {
                        console.log('Camera config failed:', e.message);
                        continue;
                    }
                }
                
                if (!success) {
                    throw new Error('All camera configurations failed');
                }
                
                video.srcObject = stream;
                video.style.display = 'block';
                
                await new Promise((resolve) => {
                    video.onloadedmetadata = () => {
                        video.play();
                        resolve();
                    };
                });
                
                showStatus('Camera started successfully!', 'success');
                
            } catch (err) {
                console.error('Error accessing camera:', err);
                showStatus(`Camera error: ${err.message}. Try the Upload Photo method instead!`, 'error');
            }
        }
        
        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
                video.style.display = 'block';
                showStatus('Camera stopped', 'success');
            }
        }
        
        async function captureImage() {
            if (!stream) {
                showStatus('Please start camera first!', 'error');
                return;
            }
            
            try {
                showStatus('Analyzing image...', 'loading');
                
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0);
                
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: imageData
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('prediction').textContent = result.prediction;
                    document.getElementById('confidence').textContent = 
                        `Confidence: ${result.confidence.toFixed(1)}%`;
                    document.getElementById('timestamp').textContent = 
                        `Analyzed at: ${new Date().toLocaleTimeString()}`;
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('status').style.display = 'none';
                } else {
                    showStatus('Prediction failed: ' + result.error, 'error');
                }
                
            } catch (error) {
                console.error('Error:', error);
                showStatus('Error processing image', 'error');
            }
        }
        
        // Initialize upload as default method
        window.addEventListener('load', function() {
            switchMethod('upload');
        });
    </script>
</body>
</html>
"""

# Flask web application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model and classes
global_model = None
global_transform = None
global_classes = None
global_device = None

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, classes=global_classes)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image']
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Make prediction
        prediction, confidence = predict_image(
            image, global_model, global_transform, global_classes, global_device
        )
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': confidence
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def start_web_server():
    """Start the Flask web server"""
    print(f"\n{'='*60}")
    print("🌐 STARTING ENHANCED WEB INTERFACE")
    print("="*60)
    print("📱 BEST FOR IPHONE: Use the 'Upload Photo' method!")
    print("   1. Take photos with your iPhone Camera app")
    print("   2. Open Safari and go to the web interface")
    print("   3. Upload your photos for classification")
    print("\n🌐 Access URLs:")
    print("   • From this computer: http://localhost:5000")
    print("   • From iPhone/other devices: http://[YOUR_PC_IP]:5000")
    print("\n🔍 To find your PC's IP address:")
    print("   • Windows: Open Command Prompt, type 'ipconfig'")
    print("   • macOS/Linux: Open Terminal, type 'ifconfig'")
    print("   • Look for 'IPv4 Address' or 'inet' (usually 192.168.x.x)")
    print("\n✨ NEW FEATURES:")
    print("   • 📱 File upload for iPhone users")
    print("   • 🖱️ Drag & drop support")
    print("   • 📷 Live camera option (backup)")
    print("   • 🎨 Enhanced mobile-friendly interface")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("="*60)
    
    # Set global variables
    global global_model, global_transform, global_classes, global_device
    global_model = model
    global_transform = test_transform
    global_classes = train_dataset.classes
    global_device = device
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

# Start web server in a separate thread after training
def launch_web_interface():
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    return web_thread

# Launch web interface
print(f"\n🚀 Launching enhanced web interface...")
web_thread = launch_web_interface()

# Keep the main thread alive
try:
    print("\n✅ Training complete! Enhanced web interface is running...")
    print("📱 iPhone users: Use the 'Upload Photo' method for best results!")
    web_thread.join()
except KeyboardInterrupt:
    print("\n🛑 Shutting down...")