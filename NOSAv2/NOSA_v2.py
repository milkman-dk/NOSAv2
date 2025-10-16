import torch
import numpy as np
import mat73
from NOSA_v2_def import UNet

def predict_tumor(file_path, model_path = 'C:/Matura/MaturaArbeitML25/NOSAv2/unet_brain_tumor_v1.3.pth', heatmap_path='tumor_heatmap_mask.npy', threshold=0.7, device=None):
    """
    Predicts tumor regions in brain MRI images using a trained UNet model.
    
    Arguments:
        file_path (str): Path to the .mat file containing the MRI image data
        model_path (str): Path to the trained UNet model weights (.pth file)
        heatmap_path (str): Path to the tumor heatmap mask (currently unused)
        threshold (float): Confidence threshold for binary mask prediction (0-1)
        device (torch.device): Computation device, defaults to GPU if available
    
    Returns:
        tuple: (pred_mask, test_img)
            - pred_mask (np.ndarray): Binary tumor mask (0s and 1s)
            - test_img (np.ndarray): Original normalized MRI image
    """

    # Set computation device (GPU preferred for faster inference)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the trained UNet model with pre-trained weights
    model = UNet(n_channels=1, n_classes=1, base_filters=64).to(device)  # Single channel grayscale input/output
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load trained weights
    model.eval()  # Set to evaluation mode (disables dropout, batch norm training mode)

    # 2. Load and preprocess the MRI image from MATLAB file
    mat = mat73.loadmat(file_path)  # Load .mat file (MATLAB v7.3+ format)
    cjdata = mat['cjdata']  # Access the data structure
    test_img = np.array(cjdata['image'], dtype=np.float32)  # Extract image as float32
    
    # Normalize image to [0, 1] range for consistent model input
    test_img_nrml = (test_img - test_img.min()) / (test_img.max() - test_img.min() + 1e-8)  # Add epsilon to prevent division by zero
    
    # Convert to PyTorch tensor with batch and channel dimensions [1, 1, H, W]
    input_tensor = torch.tensor(test_img_nrml, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # 3. Run inference to predict tumor regions
    with torch.no_grad():  # Disable gradient computation for faster inference
        output = model(input_tensor)  # Forward pass through UNet, outputs logits [1, 1, H, W]
        prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()  # Convert logits to probabilities [H, W]
        pred_mask = (prob_mask > threshold).astype(np.uint8)  # Apply threshold to create binary mask

    return pred_mask, test_img

'''
Function Summary:
- Loads a trained UNet model for brain tumor segmentation
- Preprocesses MRI images from MATLAB files (.mat format)
- Generates probability and binary tumor masks
- Uses GPU acceleration when available for faster processing
- Returns both the binary prediction mask and original image for visualization

Usage:
    pred_mask, original_img = predict_tumor("path/to/mri.mat")
    # pred_mask: Binary mask where 1 = tumor, 0 = healthy tissue
    # original_img: Original MRI image for overlay visualization
'''