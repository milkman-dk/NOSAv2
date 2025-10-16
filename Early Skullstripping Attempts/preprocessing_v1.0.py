import os
import cv2
import numpy as np
import hashlib
from PIL import Image
from skimage import exposure
import shutil
from pathlib import Path

def setup_directory_structure(base_dir):
    """Create dataset directory structure"""
    Path(base_dir).mkdir(exist_ok=True)
    dirs = {
        'raw_tumors': os.path.join(base_dir, 'raw_tumors'),
        'raw_normal': os.path.join(base_dir, 'raw_normal'),
        'processed': os.path.join(base_dir, 'processed'),
        'processed_images': os.path.join(base_dir, 'processed', 'images'),
        'processed_labels': os.path.join(base_dir, 'processed', 'labels')
    }
    
    for d in dirs.values():
        Path(d).mkdir(parents=True, exist_ok=True)
    
    return dirs

#Check if image is corrupted
def is_corrupted(image_path):
    
    try:
        img = Image.open(image_path)
        img.verify()
        return False
    except (IOError, SyntaxError):
        return True

#Identify duplicate images using perceptual hashing
def find_duplicates(image_dir):
    
    hashes = {}
    duplicates = []
    
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                with open(img_path, 'rb') as f:
                    img_hash = hashlib.md5(f.read()).hexdigest()
                if img_hash in hashes:
                    duplicates.append(img_path)
                else:
                    hashes[img_hash] = img_path
            except:
                continue
    return duplicates

#Ensure every image has a corresponding label file
def validate_label_files(image_dir, label_dir):
    
    missing_labels = []
    
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(img_file)[0]
            label_file = os.path.join(label_dir, f"{base_name}.txt")
            if not os.path.exists(label_file):
                missing_labels.append(img_file)
    
    return missing_labels

#Efficient skull stripping using adaptive thresholding
def skull_stripping(img):
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find largest contour (brain tissue)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [largest_contour], -1, (255,), -1)
        return cv2.bitwise_and(img, img, mask=mask)
    return img

#Optimized contrast enhancement using CLAHE
def enhance_contrast(img):
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def apply_filters(img):
    # Denoising
    denoised = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Edge enhancement
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened

#Resize image to target size with aspect ratio preservation
def resize_with_padding(img, target_size=640):
    
    h, w = img.shape
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create padded image
    padded = np.zeros((target_size, target_size), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded, (x_offset, y_offset, new_w, new_h)

#Complete MRI preprocessing pipeline for YOLOv9
def preprocess_mri_pipeline(base_dir):
    
    dirs = setup_directory_structure(base_dir)
    
    # Process tumor images
    process_images(dirs['raw_tumors'], dirs['processed_images'], dirs['processed_labels'], has_tumor=True)
    
    # Process normal images
    process_images(dirs['raw_normal'], dirs['processed_images'], dirs['processed_labels'], has_tumor=False)
    
    print("Preprocessing completed successfully!")

def process_images(src_dir, dest_img_dir, dest_label_dir, has_tumor=True):
    """Process individual images with tumor/normal handling"""
    # Step 1: Remove corrupted images
    for img_file in os.listdir(src_dir):
        img_path = os.path.join(src_dir, img_file)
        if is_corrupted(img_path):
            print(f"Removing corrupted image: {img_file}")
            os.remove(img_path)
    
    # Step 2: Remove duplicates
    duplicates = find_duplicates(src_dir)
    for dup in duplicates:
        print(f"Removing duplicate: {os.path.basename(dup)}")
        os.remove(dup)
    
    # Step 3: Validate label files for tumor images
    if has_tumor:
        missing = validate_label_files(src_dir, src_dir)
        if missing:
            print(f"Missing labels for {len(missing)} tumor images. Processing aborted.")
            return
    
    # Process each image
    for img_file in os.listdir(src_dir):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(src_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Skip if image couldn't be loaded
        if img is None:
            continue
            
        # Processing pipeline
        #img = skull_stripping(img)
        img = enhance_contrast(img)
        img = apply_filters(img)
        img, padding_info = resize_with_padding(img)
        
        # Save processed image
        output_img_path = os.path.join(dest_img_dir, img_file)
        cv2.imwrite(output_img_path, img)
        
        # Handle label files for tumor images
        if has_tumor:
            base_name = os.path.splitext(img_file)[0]
            label_path = os.path.join(src_dir, f"{base_name}.txt")
            output_label_path = os.path.join(dest_label_dir, f"{base_name}.txt")
            
            # Update bounding box coordinates for resized/padded image
            update_bounding_boxes(label_path, output_label_path, img.shape, padding_info)

#Adjust bounding boxes for resized/padded images
def update_bounding_boxes(input_label, output_label, img_shape, padding_info):
    
    x_offset, y_offset, new_w, new_h = padding_info
    target_size = img_shape[0]  # 640x640
    
    with open(input_label, 'r') as f_in, open(output_label, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            class_id, x_center, y_center, width, height = map(float, parts)
            
            # Convert to absolute coordinates
            orig_w, orig_h = 1, 1  # Normalized coordinates
            abs_x = x_center * orig_w
            abs_y = y_center * orig_h
            abs_w = width * orig_w
            abs_h = height * orig_h
            
            # Scale to resized dimensions
            scale = target_size / max(orig_w, orig_h)
            new_x = abs_x * scale
            new_y = abs_y * scale
            new_w = abs_w * scale
            new_h = abs_h * scale
            
            # Adjust for padding
            new_x += x_offset
            new_y += y_offset
            
            # Convert back to normalized coordinates
            norm_x = new_x / target_size
            norm_y = new_y / target_size
            norm_w = new_w / target_size
            norm_h = new_h / target_size
            
            f_out.write(f"{int(class_id)} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

if __name__ == "__main__":
    BASE_DIR = "C:/Matura/TEST"
    preprocess_mri_pipeline(BASE_DIR)
