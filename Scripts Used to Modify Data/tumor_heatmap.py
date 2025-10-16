import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# Path to folder
folder_path = 'C:/Matura/data'

# Target size
target_shape = (512, 512)

# Initialize accumulator
heatmap = np.zeros(target_shape, dtype=np.float32)
file_count = 0

# Loop through all .mat files
for filename in os.listdir(folder_path):
    if filename.endswith('.mat'):
        file_path = os.path.join(folder_path, filename)
        with h5py.File(file_path, 'r') as f:
            mask = np.array(f['cjdata/tumorMask'])
            
            # Resize mask to 512x512
            resized_mask = resize(mask, target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
            
            # Accumulate
            heatmap += resized_mask
            file_count += 1

# Normalize heatmap
if file_count > 0 and heatmap.max() != 0:
    heatmap /= heatmap.max()

# Save heatmap
if file_count > 0:
    np.save('tumor_heatmap_mask.npy', heatmap)
    print(f"Heatmap saved to 'tumor_heatmap_mask.npy' with {file_count} files processed.")
else:
    print("No heatmap data to save.")

# Display heatmap
if file_count > 0:
    plt.imshow(heatmap, cmap='hot')
    plt.colorbar(label='Normalized Overlap')
    plt.title('Tumor Mask Heatmap')
    plt.show()
else:
    print("No heatmap data to display.")