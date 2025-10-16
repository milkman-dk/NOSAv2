import cv2
import numpy as np
from skimage import measure, morphology

def adaptive_gamma_correction(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                     for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def skull_stripping_mri(img_path):
    # 1. Load image and convert to grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Adaptive Gamma Correction
    mean_intensity = np.mean(gray)
    gamma = 0.5 if mean_intensity < 100 else 2.0
    enhanced = adaptive_gamma_correction(gray, gamma)
    
    # 3. Corrected Otsu Thresholding (INVERTED)
    _, thresh = cv2.threshold(enhanced, 0, 255, 
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Find largest connected component (brain)
    labels = measure.label(thresh)
    regions = measure.regionprops(labels)
    if not regions:
        return img  # Fallback if no regions found
    
    # Sort regions by area (descending) and skip background
    regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)
    brain_region = regions_sorted[0]  # Largest region
    
    # 5. Create brain mask
    brain_mask = np.zeros_like(thresh)
    brain_mask[labels == brain_region.label] = 255
    
    # 6. Morphological refinement
    closed = morphology.closing(brain_mask, morphology.disk(5))
    contours, _ = cv2.findContours(closed.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    
    # Fill largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    filled_mask = np.zeros_like(closed)
    cv2.drawContours(filled_mask, [largest_contour], -1, 255, cv2.FILLED)
    
    # 7. Final brain extraction
    result = cv2.bitwise_and(img, img, mask=filled_mask)
    return result


stripped_image = skull_stripping_mri("C:/Matura/TEST/raw_tumors/tumor_00007.jpg")
cv2.imwrite("stripped_brain.jpg", stripped_image)