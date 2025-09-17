import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from PIL import Image
import os
import cv2
import warnings
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from collections import Counter

warnings.filterwarnings('ignore')

# Start timer and setup
start_time = time.time()
print("="*80)
print("=== GTSDB HYBRID PIPELINE: EVALUATION SCRIPT ===")
print("="*80)

# CUDA Setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# GTSDB Parser class (needed for loading training metadata)
class GTSDBParser:
    def __init__(self, annotation_file):
        self.annotations = {}
        self.filename_mapping = {}
        self.class_counts = Counter()
        
        if not os.path.exists(annotation_file):
            print(f"Annotation file not found: {annotation_file}")
            return
            
        with open(annotation_file, 'r') as f:
            for line_num, line in enumerate(f):
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split(';')
                    if len(parts) >= 6:
                        filename = parts[0]
                        try:
                            x1, y1, x2, y2 = map(int, parts[1:5])
                            original_class_id = int(parts[5])
                            
                            self.class_counts[original_class_id] += 1
                            
                            base_name = os.path.splitext(filename)[0]
                            self.filename_mapping[filename] = filename
                            self.filename_mapping[base_name + '.png'] = filename
                            self.filename_mapping[base_name + '.jpg'] = filename
                            
                            if filename not in self.annotations:
                                self.annotations[filename] = []
                            self.annotations[filename].append({
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'original_class_id': original_class_id,
                                'yolo_class_id': 0
                            })
                        except ValueError:
                            continue
    
    def get_annotations(self, filename):
        if filename in self.annotations:
            return self.annotations[filename]
        
        if filename in self.filename_mapping:
            original_name = self.filename_mapping[filename]
            return self.annotations.get(original_name, [])
        
        base_name = os.path.splitext(filename)[0]
        for ext in ['.ppm', '.png', '.jpg', '.jpeg']:
            test_name = base_name + ext
            if test_name in self.annotations:
                return self.annotations[test_name]
        
        return []

# Load training metadata
print("Loading training metadata...")
training_metadata_file = r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results\training_metadata.pkl'

if not os.path.exists(training_metadata_file):
    print(f"ERROR: Training metadata not found at {training_metadata_file}")
    print("Please run the training script first!")
    exit(1)

with open(training_metadata_file, 'rb') as f:
    training_metadata = pickle.load(f)

# Extract training information
trained_weights = training_metadata['trained_weights']
yolo_dataset_path = training_metadata['yolo_dataset_path']
test_parser = training_metadata['test_parser']
dataset_stats = training_metadata['dataset_stats']

print(f"Loaded training metadata:")
print(f"  - Trained weights: {trained_weights}")
print(f"  - Dataset path: {yolo_dataset_path}")
print(f"  - Test images: {dataset_stats['test_count']}")

# Check if trained model exists
if not os.path.exists(trained_weights):
    print(f"ERROR: Trained weights not found at {trained_weights}")
    print("Please check if training completed successfully!")
    exit(1)

# Load YOLO model
print("Loading YOLO model...")
try:
    from ultralytics import YOLO
    model = YOLO(trained_weights)
    if hasattr(model.model, 'to'):
        model.model.to(device)
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"ERROR loading YOLO model: {e}")
    exit(1)

# Load GTSRB Classification Model
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

if gtsrb_model is None:
    print("WARNING: GTSRB model not available - will only evaluate detection performance")

# Helper functions
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x1 < x2 and y1 < y2:
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0
    return 0

def match_predictions_to_gt(predictions, ground_truth, iou_threshold=0.5):
    """Match predictions to ground truth based on IoU"""
    matched_pairs = []
    used_gt = set()
    
    for pred in predictions:
        pred_box = [pred['x1'], pred['y1'], pred['x2'], pred['y2']]
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in used_gt:
                continue
                
            gt_box = [gt['x1'], gt['y1'], gt['x2'], gt['y2']]
            iou = calculate_iou(pred_box, gt_box)
            
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx >= 0:
            matched_pairs.append({
                'prediction': pred,
                'ground_truth': ground_truth[best_gt_idx],
                'iou': best_iou
            })
            used_gt.add(best_gt_idx)
    
    return matched_pairs

# ===========================================================================================
# HYBRID PIPELINE EVALUATION
# ===========================================================================================
print("\n" + "="*80)
print("=== HYBRID PIPELINE EVALUATION ===")
print("="*80)

test_images_path = os.path.join(yolo_dataset_path, 'images', 'test')
test_files = [f for f in os.listdir(test_images_path) if f.endswith(('.ppm', '.jpg', '.png'))]

print(f"Evaluating hybrid pipeline on {len(test_files)} test images...")

# Results storage
evaluation_results = {
    'detection_times': [],
    'classification_times': [],
    'total_pipeline_times': [],
    'detection_confidences': [],
    'classification_confidences': [],
    'detection_results': [],
    'classification_results': [],
    'matched_pairs': []
}

detected_count = 0
gt_count = 0
files_with_gt = 0

# Binary metrics for detection
y_true_detection = []
y_pred_detection = []

# Classification metrics
y_true_classification = []
y_pred_classification = []

evaluation_start = time.time()

# Warmup
print("Warming up models...")
for i, filename in enumerate(test_files[:min(5, len(test_files))]):
    img_path = os.path.join(test_images_path, filename)
    try:
        _ = model(img_path, conf=0.4, verbose=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    except Exception:
        continue

max_samples = min(500, len(test_files))
print(f"Processing {max_samples} test images...")

# Create debug directory
debug_dir = r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results\debug_patches'
os.makedirs(debug_dir, exist_ok=True)
debug_counter = 0
max_debug_patches = 20  # Limit debug patches to avoid too many files

for idx, filename in enumerate(test_files[:max_samples]):
    img_path = os.path.join(test_images_path, filename)
    
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]
    
    # Ground truth
    gt_boxes = test_parser.get_annotations(filename)
    if len(gt_boxes) > 0:
        files_with_gt += 1
        gt_count += len(gt_boxes)
        y_true_detection.append(1)
    else:
        y_true_detection.append(0)
    
    # Phase 1: YOLO Detection
    start_detection = time.time()
    try:
        yolo_results = model(img_path, conf=0.4, verbose=False)
        detection_time = (time.time() - start_detection) * 1000
    except Exception:
        continue
    
    evaluation_results['detection_times'].append(detection_time)
    
    # Process YOLO detections
    detections = []
    
    for result in yolo_results:
        if result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                detections.append({
                    'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                    'confidence': float(conf)
                })
                evaluation_results['detection_confidences'].append(float(conf))
                detected_count += 1
    
    # Binary detection result
    y_pred_detection.append(1 if len(detections) > 0 else 0)
    
    # Phase 2: GTSRB Classification (if model available)
    start_classification = time.time()
    classification_results = []
    
    if gtsrb_model is not None and len(detections) > 0:
        for det_idx, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
            
            # Debug: Print detection coordinates
            if debug_counter < max_debug_patches:
                print(f"Debug patch {debug_counter}: Detection at ({x1},{y1},{x2},{y2}), Image size: {img_width}x{img_height}")
            
            # Extract patch
            if 0 <= y1 < y2 <= img_height and 0 <= x1 < x2 <= img_width:
                patch = img_rgb[y1:y2, x1:x2]
                
                if patch.size > 0:
                    # Debug: Save patch and print info
                    if debug_counter < max_debug_patches:
                        patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                        debug_filename = f'patch_{debug_counter}_{filename}_{det_idx}_det{detection["confidence"]:.3f}.jpg'
                        debug_path = os.path.join(debug_dir, debug_filename)
                        cv2.imwrite(debug_path, patch_bgr)
                        print(f"  Saved debug patch: {debug_filename}")
                        print(f"  Patch shape: {patch.shape}")
                        print(f"  Detection confidence: {detection['confidence']:.3f}")
                        
                        # Also save corresponding GT patch if available
                        if len(gt_boxes) > 0:
                            gt_box = gt_boxes[0]  # Take first GT box for comparison
                            gt_x1, gt_y1, gt_x2, gt_y2 = gt_box['x1'], gt_box['y1'], gt_box['x2'], gt_box['y2']
                            if 0 <= gt_y1 < gt_y2 <= img_height and 0 <= gt_x1 < gt_x2 <= img_width:
                                gt_patch = img_rgb[gt_y1:gt_y2, gt_x1:gt_x2]
                                if gt_patch.size > 0:
                                    gt_patch_bgr = cv2.cvtColor(gt_patch, cv2.COLOR_RGB2BGR)
                                    gt_debug_filename = f'patch_{debug_counter}_{filename}_{det_idx}_GT_class{gt_box["original_class_id"]}.jpg'
                                    gt_debug_path = os.path.join(debug_dir, gt_debug_filename)
                                    cv2.imwrite(gt_debug_path, gt_patch_bgr)
                                    print(f"  Saved GT patch: {gt_debug_filename}")
                                    print(f"  GT patch shape: {gt_patch.shape}")
                                    print(f"  GT class: {gt_box['original_class_id']}")
                        
                        debug_counter += 1
                    
                    try:
                        patch_pil = Image.fromarray(patch)
                        patch_tensor = gtsrb_transform(patch_pil).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            class_output = gtsrb_model(patch_tensor)
                            class_probs = torch.softmax(class_output, dim=1)
                            class_conf, class_pred = torch.max(class_probs, 1)
                        
                        classification_results.append({
                            'detection': detection,
                            'predicted_class': class_pred.item(),
                            'confidence': class_conf.item()
                        })
                        
                        # Debug: Print classification result
                        if debug_counter <= max_debug_patches:
                            print(f"  Predicted class: {class_pred.item()}, confidence: {class_conf.item():.3f}")
                        
                        evaluation_results['classification_confidences'].append(class_conf.item())
                        
                    except Exception as e:
                        if debug_counter <= max_debug_patches:
                            print(f"  Classification error: {e}")
                        continue
                else:
                    if debug_counter <= max_debug_patches:
                        print(f"  Empty patch detected")
            else:
                if debug_counter <= max_debug_patches:
                    print(f"  Invalid coordinates: ({x1},{y1},{x2},{y2}) for image {img_width}x{img_height}")
    
    classification_time = (time.time() - start_classification) * 1000
    evaluation_results['classification_times'].append(classification_time)
    evaluation_results['total_pipeline_times'].append(detection_time + classification_time)
    
    # Match predictions to ground truth for classification evaluation
    if len(classification_results) > 0 and len(gt_boxes) > 0:
        # Convert classification results to detection format for matching
        pred_detections = [cr['detection'] for cr in classification_results]
        matched_pairs = match_predictions_to_gt(pred_detections, gt_boxes, iou_threshold=0.5)
        
        for pair in matched_pairs:
            # Find corresponding classification result
            pred_detection = pair['prediction']
            for cr in classification_results:
                if cr['detection'] == pred_detection:
                    y_true_classification.append(pair['ground_truth']['original_class_id'])
                    y_pred_classification.append(cr['predicted_class'])
                    break
        
        evaluation_results['matched_pairs'].extend(matched_pairs)
    
    evaluation_results['detection_results'].append(detections)
    evaluation_results['classification_results'].append(classification_results)
    
    if (idx + 1) % 100 == 0:
        print(f"Evaluation progress: {idx + 1}/{max_samples} ({((idx + 1)/max_samples*100):.1f}%)")

evaluation_time = time.time() - evaluation_start

# Calculate metrics
print(f"Evaluation completed in {evaluation_time:.1f} seconds")

# Detection metrics (binary)
detection_accuracy = accuracy_score(y_true_detection, y_pred_detection)
detection_precision, detection_recall, detection_f1, _ = precision_recall_fscore_support(
    y_true_detection, y_pred_detection, average='binary', zero_division=0
)

# Classification metrics (if available)
if len(y_true_classification) > 0:
    classification_accuracy = accuracy_score(y_true_classification, y_pred_classification)
    classification_precision, classification_recall, classification_f1, _ = precision_recall_fscore_support(
        y_true_classification, y_pred_classification, average='weighted', zero_division=0
    )
else:
    classification_accuracy = 0.0
    classification_precision = 0.0
    classification_recall = 0.0
    classification_f1 = 0.0

# ===========================================================================================
# PERFORMANCE ANALYSIS + RESULTS
# ===========================================================================================
print("\n" + "="*80)
print("=== PERFORMANCE ANALYSIS + RESULTS ===")
print("="*80)

# Detection Performance
if len(evaluation_results['detection_times']) > 0:
    detection_times = evaluation_results['detection_times']
    
    mean_detection = np.mean(detection_times)
    std_detection = np.std(detection_times)
    min_detection = np.min(detection_times)
    max_detection = np.max(detection_times)
    
    print("DETECTION PERFORMANCE:")
    print(f"  Mittlere Latenz:           {mean_detection:.2f} ms")
    print(f"  Standardabweichung:        {std_detection:.2f} ms")
    print(f"  Min Latenz:                {min_detection:.2f} ms")
    print(f"  Max Latenz:                {max_detection:.2f} ms")

# Classification Performance  
if len(evaluation_results['classification_times']) > 0:
    classification_times = evaluation_results['classification_times']
    
    mean_classification = np.mean(classification_times)
    std_classification = np.std(classification_times)
    
    print(f"\nCLASSIFICATION PERFORMANCE:")
    print(f"  Mittlere Latenz:           {mean_classification:.2f} ms")
    print(f"  Standardabweichung:        {std_classification:.2f} ms")

# Pipeline Performance
if len(evaluation_results['total_pipeline_times']) > 0:
    total_times = evaluation_results['total_pipeline_times']
    
    mean_total = np.mean(total_times)
    pipeline_fps = 1000/mean_total if mean_total > 0 else 0
    
    print(f"\nPIPELINE PERFORMANCE:")
    print(f"  Mittlere Gesamt-Latenz:    {mean_total:.2f} ms")
    print(f"  Pipeline Durchsatz:        {pipeline_fps:.1f} FPS")
    
    # Latency distribution
    fast_count = len([t for t in total_times if t <= 30])
    medium_count = len([t for t in total_times if 30 < t <= 60])
    slow_count = len([t for t in total_times if t > 60])
    
    print(f"  Latenz-Verteilung:")
    print(f"    ≤30ms:                   {fast_count} ({fast_count/len(total_times)*100:.1f}%)")
    print(f"    30-60ms:                 {medium_count} ({medium_count/len(total_times)*100:.1f}%)")
    print(f"    >60ms:                   {slow_count} ({slow_count/len(total_times)*100:.1f}%)")

# Confidence Analysis
if evaluation_results['detection_confidences']:
    det_confidences = evaluation_results['detection_confidences']
    
    mean_det_conf = np.mean(det_confidences)
    std_det_conf = np.std(det_confidences)
    min_det_conf = np.min(det_confidences)
    max_det_conf = np.max(det_confidences)
    
    print(f"\nDETECTION CONFIDENCE:")
    print(f"  Mittlere Konfidenz:        {mean_det_conf:.3f}")
    print(f"  Standardabweichung:        {std_det_conf:.3f}")
    print(f"  Min Konfidenz:             {min_det_conf:.3f}")
    print(f"  Max Konfidenz:             {max_det_conf:.3f}")

if evaluation_results.get('classification_confidences'):
    cls_confidences = evaluation_results['classification_confidences']
    
    mean_cls_conf = np.mean(cls_confidences)
    
    print(f"\nCLASSIFICATION CONFIDENCE:")
    print(f"  Mittlere Konfidenz:        {mean_cls_conf:.3f}")

# Results Summary
print(f"\nERGEBNISSE:")
print(f"  Verarbeitete Testbilder:   {len(evaluation_results['detection_times'])}")
print(f"  Dateien mit GT-Daten:      {files_with_gt}")
print(f"  Detektierte Objekte:       {detected_count}")
print(f"  Ground Truth Objekte:      {gt_count}")
if gt_count > 0:
    detection_rate = detected_count / gt_count * 100
    print(f"  Detection Rate:            {detection_rate:.1f}%")

print(f"  Erfolgreich zugeordnet:    {len(y_true_classification)}")
if len(evaluation_results['matched_pairs']) > 0:
    total_detections = sum(len(dr) for dr in evaluation_results['detection_results'])
    matching_rate = len(evaluation_results['matched_pairs']) / total_detections * 100 if total_detections > 0 else 0
    print(f"  Matching Rate (IoU≥0.5):   {matching_rate:.1f}%")

# Detection Metrics
print(f"\nDETECTION METRIKEN:")
print(f"  Accuracy:                  {detection_accuracy:.4f}")
print(f"  Precision:                 {detection_precision:.4f}")
print(f"  Recall:                    {detection_recall:.4f}")
print(f"  F1-Score:                  {detection_f1:.4f}")

# Classification Metrics
print(f"\nCLASSIFICATION METRIKEN:")
if len(y_true_classification) > 0:
    print(f"  Accuracy:                  {classification_accuracy:.4f}")
    print(f"  Precision:                 {classification_precision:.4f}")
    print(f"  Recall:                    {classification_recall:.4f}")
    print(f"  F1-Score:                  {classification_f1:.4f}")
    print(f"  Klassifizierte Samples:    {len(y_true_classification)}")
else:
    print(f"  Keine erfolgreiche Klassifikation möglich")

# Class distribution analysis (if classification available)
if len(y_true_classification) > 0:
    gt_class_counts = Counter(y_true_classification)
    pred_class_counts = Counter(y_pred_classification)
    
    print(f"\nTOP 10 GROUND TRUTH CLASSES:")
    for i, (class_id, count) in enumerate(gt_class_counts.most_common(10), 1):
        percentage = count / len(y_true_classification) * 100
        print(f"  {i:2d}. Class {class_id:2d}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\nTOP 10 PREDICTED CLASSES:")
    for i, (class_id, count) in enumerate(pred_class_counts.most_common(10), 1):
        percentage = count / len(y_pred_classification) * 100
        print(f"  {i:2d}. Class {class_id:2d}: {count:4d} ({percentage:5.1f}%)")

# ===========================================================================================
# FINAL EVALUATION SUMMARY
# ===========================================================================================
print("\n" + "="*80)
print("=== FINAL EVALUATION SUMMARY ===")
print("="*80)

total_time = time.time() - start_time

print(f"Evaluation completed successfully.")
print(f"Total evaluation runtime: {total_time/60:.2f} minutes")

print(f"\nCORE METRICS:")
if len(evaluation_results['detection_times']) > 0:
    detection_fps = 1000/np.mean(evaluation_results['detection_times'])
    print(f"  Detection Durchsatz:       {detection_fps:.1f} FPS")
    print(f"  Detection Latenz:          {np.mean(evaluation_results['detection_times']):.2f} ms")
    if evaluation_results['detection_confidences']:
        print(f"  Detection Konfidenz:       {np.mean(evaluation_results['detection_confidences']):.3f}")

if len(evaluation_results['total_pipeline_times']) > 0:
    print(f"  Pipeline Durchsatz:        {pipeline_fps:.1f} FPS")
    print(f"  Pipeline Latenz:           {np.mean(evaluation_results['total_pipeline_times']):.2f} ms")

print(f"\nDETECTION PERFORMANCE:")
print(f"  Accuracy:                  {detection_accuracy:.4f}")
print(f"  Precision:                 {detection_precision:.4f}")
print(f"  Recall:                    {detection_recall:.4f}")
print(f"  F1-Score:                  {detection_f1:.4f}")

if len(y_true_classification) > 0:
    print(f"\nCLASSIFICATION PERFORMANCE:")
    print(f"  Accuracy:                  {classification_accuracy:.4f}")
    print(f"  Precision:                 {classification_precision:.4f}")
    print(f"  Recall:                    {classification_recall:.4f}")
    print(f"  F1-Score:                  {classification_f1:.4f}")
    print(f"  Klassifizierte Samples:    {len(y_true_classification)}")
else:
    print(f"\nCLASSIFICATION PERFORMANCE:")
    print(f"  Keine Klassifikation durchgeführt (GTSRB-Modell nicht verfügbar)")

# ===========================================================================================
# SAVE EVALUATION RESULTS
# ===========================================================================================
print(f"\nSaving evaluation results...")

# Create results directory if it doesn't exist
os.makedirs(r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results', exist_ok=True)

# Define file paths
evaluation_metadata_file = r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results\evaluation_metadata.pkl'
predictions_txt_file = r'G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results\gtsdb_hybrid_pipeline_predictions.txt'

# Save comprehensive evaluation results
evaluation_metadata = {
    'trained_weights': trained_weights,
    'yolo_dataset_path': yolo_dataset_path,
    'test_parser': test_parser,
    'start_time_evaluation': start_time,
    'device': str(device),
    'evaluation_results': evaluation_results,
    'cuda_available': torch.cuda.is_available(),
    'files_with_gt': files_with_gt,
    'detected_count': detected_count,
    'gt_count': gt_count,
    'detection_accuracy': detection_accuracy,
    'detection_precision': detection_precision,
    'detection_recall': detection_recall,
    'detection_f1': detection_f1,
    'classification_accuracy': classification_accuracy,
    'classification_precision': classification_precision,
    'classification_recall': classification_recall,
    'classification_f1': classification_f1,
    'evaluation_time': evaluation_time,
    'gtsrb_model_available': gtsrb_model is not None,
    'classification_samples': len(y_true_classification),
    'training_metadata': training_metadata  # Include training info
}

with open(evaluation_metadata_file, 'wb') as f:
    pickle.dump(evaluation_metadata, f)

# Save detailed predictions to TXT file
with open(predictions_txt_file, 'w') as f:
    f.write("# GTSDB Hybrid Pipeline Predictions\n")
    f.write("# Format: filename,detection_x1,detection_y1,detection_x2,detection_y2,detection_conf,predicted_class,class_conf,gt_class,iou\n")
    
    for idx, filename in enumerate(test_files[:max_samples]):
        if idx < len(evaluation_results['detection_results']):
            detections = evaluation_results['detection_results'][idx]
            classifications = evaluation_results['classification_results'][idx] if idx < len(evaluation_results['classification_results']) else []
            
            # Write detection results
            for det in detections:
                line_parts = [
                    filename,
                    str(det['x1']), str(det['y1']), str(det['x2']), str(det['y2']),
                    f"{det['confidence']:.3f}"
                ]
                
                # Find corresponding classification
                predicted_class = -1
                class_conf = 0.0
                for cls in classifications:
                    if cls['detection'] == det:
                        predicted_class = cls['predicted_class']
                        class_conf = cls['confidence']
                        break
                
                line_parts.extend([str(predicted_class), f"{class_conf:.3f}"])
                
                # Find corresponding GT and IoU
                gt_class = -1
                iou = 0.0
                for pair in evaluation_results['matched_pairs']:
                    if pair['prediction'] == det:
                        gt_class = pair['ground_truth']['original_class_id']
                        iou = pair['iou']
                        break
                
                line_parts.extend([str(gt_class), f"{iou:.3f}"])
                
                f.write(",".join(line_parts) + "\n")

print(f"\nSAVED FILES:")
print(f"  - {evaluation_metadata_file}")
print(f"  - {predictions_txt_file}")

print(f"\n" + "="*80)
print("=== EVALUATION COMPLETED SUCCESSFULLY ===")
print("="*80)