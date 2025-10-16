import cv2
import numpy as np
from skimage import morphology, measure

def adaptive_gamma_correction(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                     for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def skull_stripping_mri(
    img_path,
    output_path=None,
    erosion_size=7,         # Kernel size for erosion (disconnects brain/skull)
    closing_size=10,        # Kernel size for closing (removes skull fragments)
    clahe_clip=5.0,         # CLAHE clip limit (contrast enhancement)
    clahe_tile=(4,4),       # CLAHE tile grid size
    gamma_dark=0.3,         # Gamma for dark images
    gamma_bright=2.5        # Gamma for bright images
):
    # 1. Load image and convert to grayscale
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    enhanced = clahe.apply(gray)
    
    # 3. Adaptive gamma correction
    mean_intensity = np.mean(enhanced)
    gamma = gamma_dark if mean_intensity < 100 else gamma_bright
    adjusted = adaptive_gamma_correction(enhanced, gamma)
    
    # 4. Otsu thresholding (inverted)
    _, thresh = cv2.threshold(adjusted, 0, 255, 
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 5. Erosion to disconnect brain from skull
    kernel_erode = np.ones((erosion_size, erosion_size), np.uint8)
    eroded = cv2.erode(thresh, kernel_erode)
    
    # 6. Morphological closing to fill holes and remove small skull fragments
    closed = morphology.closing(eroded, morphology.disk(closing_size))
    closed = (closed * 255).astype(np.uint8) if closed.max() <= 1 else closed

    # 7. Find largest connected component (brain)
    labels = measure.label(closed)
    regions = measure.regionprops(labels)
    if not regions:
        print("No regions found, returning original image.")
        return img
    brain_region = max(regions, key=lambda r: r.area)
    brain_mask = np.zeros_like(closed)
    brain_mask[labels == brain_region.label] = 255

    # 8. Fill largest contour to ensure solid brain mask
    contours, _ = cv2.findContours(brain_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(brain_mask)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled_mask, [largest_contour], -1, (255,), cv2.FILLED)
    else:
        filled_mask = brain_mask

    # 9. Final brain extraction
    result = cv2.bitwise_and(img, img, mask=filled_mask.astype(np.uint8))
    
    # 10. Optionally save the result
    if output_path is not None:
        cv2.imwrite(output_path, result)
    
    return result


if __name__ == "__main__":
    result = skull_stripping_mri(
        "C:/Matura/TEST/raw_tumors/tumor_00007.jpg",
        output_path="stripped_brain_004.jpg",
        erosion_size=11,         # Try 7-11 for more/less aggressive
        closing_size=15,        # Try 10-15 for more/less aggressive
        clahe_clip=10.0,         # Try 5.0-8.0 for more/less contrast
        clahe_tile=(4,4),       # Smaller tiles = more local contrast
        gamma_dark=0.3,
        gamma_bright=3.0
    )
    # To display the result:

cv2.imshow("Stripped Brain 004", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
