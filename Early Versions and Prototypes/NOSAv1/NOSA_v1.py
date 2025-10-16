import torch
import numpy as np
import mat73
import os
from NOSA_def import UNet

def predict_tumor(file_path, model_path='c:/Matura/MaturaArbeitML25/NOSA/unet_brain_tumor_v1.0.pth', heatmap_path='tumor_heatmap_mask.npy', threshold=0.45, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the trained model
    model = UNet(n_channels=1, n_classes=1, base_filters=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load and preprocess the test image
    mat = mat73.loadmat(file_path)
    cjdata = mat['cjdata']
    test_img = np.array(cjdata['image'], dtype=np.float32)
    test_img_nrml = (test_img - test_img.min()) / (test_img.max() - test_img.min() + 1e-8)  # Normalize image
    input_tensor = torch.tensor(test_img_nrml, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # 4. Predict the mask and confidence
    with torch.no_grad():
        output = model(input_tensor)  # [1, 1, H, W]
        prob_mask = output.squeeze().cpu().numpy()  # [H, W], values in [0, 1]
        pred_mask = (prob_mask > threshold).astype(np.uint8)  # Binary mask

    return pred_mask, test_img

'''
Args:
    file_path (str): Path to the .mat file containing the test image and real mask.
    model_path (str): Path to the trained UNet model.
    heatmap_path (str): Path to the tumor heatmap mask.
    threshold (float): Threshold for binary mask prediction.
    device (torch.device): Device to run the model on, defaults to GPU if available.
Returns:
    pred_mask (np.ndarray): Predicted binary mask of the tumor.
    prob_mask (np.ndarray): Probability mask of the tumor.
        
'''