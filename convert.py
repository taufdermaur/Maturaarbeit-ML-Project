from PIL import Image
import os

folder = r"G:\Meine Ablage\Maturaarbeit\Resultate\GTSDB_YOLO_Results\debug_patches"
for fname in os.listdir(folder):
    if fname.endswith('.jpg'):
        img_path = os.path.join(folder, fname)
        img = Image.open(img_path)
        png_path = img_path.replace('.jpg', '.png')
        img.save(png_path)