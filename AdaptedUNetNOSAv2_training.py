# Fixed imports - use the new unified API
import os
import mat73
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.optim as optim


# Data preprocessing class for MATLAB files with basic normalization and resizing
class MatlabTumorDataset(Dataset):
    def __init__(self, mat_dir, target_size=(256, 256)):
        self.mat_files = [os.path.join(mat_dir, f) for f in os.listdir(mat_dir) if f.endswith('.mat')]
        self.target_size = target_size

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        mat = mat73.loadmat(self.mat_files[idx])
        cjdata = mat['cjdata']
        image = np.array(cjdata['image'], dtype=np.float32)
        mask = np.array(cjdata['tumorMask'], dtype=np.float32)

        # Normalize image to 0-1
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        mask = (mask > 0).astype(np.float32)  # Binary mask

        # Add channel dimension to ensure data matches expected input shape of UNet [Channel, Height, Width]
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convert to torch.Tensor before resizing
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # Resize images and masks
        image = TF.resize(image, self.target_size, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.target_size, interpolation=transforms.InterpolationMode.NEAREST)

        # Data augmentation (random horizontal flip to avoid overfitting and ge tmore data without increasing dataset size)
        if torch.rand(1) < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        return image, mask


# Double convolution block
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, base_filters=64):  # NOSAv2: 64 base filters (vs 32 in v1)
        super(UNet, self).__init__()

        # ENCODER (Contracting Path)
        self.inc = DoubleConv(n_channels, base_filters)  # Encoder Block 1
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2), 
            DoubleConv(base_filters, base_filters * 2)  # 128 filters
        )  # Encoder Block 2
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2), 
            DoubleConv(base_filters * 2, base_filters * 4)  # 256 filters
        )  # Encoder Block 3
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2), 
            DoubleConv(base_filters * 4, base_filters * 8)  # 512 filters
        )  # Encoder Block 4
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2), 
            DoubleConv(base_filters * 8, base_filters * 16)  # Bottleneck: 1024 filters (vs 256 in v1)
        )  # Bottleneck

        # DECODER (Expanding Path)
        self.up1 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)  # UpSampling 1
        self.conv1 = DoubleConv(base_filters * 8 + base_filters * 8, base_filters * 4)  # Decoder Block 1
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 4, 2, stride=2)  # UpSampling 2
        self.conv2 = DoubleConv(base_filters * 4 + base_filters * 4, base_filters * 2)  # Decoder Block 2
        self.up3 = nn.ConvTranspose2d(base_filters * 2, base_filters * 2, 2, stride=2)  # UpSampling 3
        self.conv3 = DoubleConv(base_filters * 2 + base_filters * 2, base_filters)      # Decoder Block 3
        self.up4 = nn.ConvTranspose2d(base_filters, base_filters, 2, stride=2)          # UpSampling 4
        self.conv4 = DoubleConv(base_filters + base_filters, base_filters)              # Decoder Block 4

        # OUTPUT
        self.outc = nn.Conv2d(base_filters, n_classes, 1)  # Final classification (1x1 conv)

    def forward(self, x):
    # Encoder
        x1 = self.inc(x)        # 64 filters
        x2 = self.down1(x1)     # 128 filters
        x3 = self.down2(x2)     # 256 filters
        x4 = self.down3(x3)     # 512 filters
        x5 = self.down4(x4)     # 1024 filters (bottleneck)
        # Decoder
        x = self.up1(x5)        # upsampled to 512 filters
        x4_resized = x4         
        x = self.conv1(torch.cat([x, x4_resized], dim=1))  # 1024 channels -> 256 filters
        x = self.up2(x)         # upsampled to 256 filters
        x3_resized = x3
        x = self.conv2(torch.cat([x, x3_resized], dim=1))  # 512 channels -> 128 filters
        x = self.up3(x)         # 128 filters
        x2_resized = x2
        x = self.conv3(torch.cat([x, x2_resized], dim=1))  # 256 channels -> 64 filters
        x = self.up4(x)         # 64 filters
        x1_resized = x1
        x = self.conv4(torch.cat([x, x1_resized], dim=1))  # 128 channels -> 64 filters
        x = self.outc(x)

        return x  # NOSAv2: Returns raw logits (no sigmoid)


# Combined Dice + BCE Loss Function - NOSAv2 uses advanced loss
class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        # Initialization: define BCE loss with logits for numerical stability
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        # Flatten inputs and targets to 1D vectors for element-wise operations
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # Compute BCE loss between raw network outputs (logits) and ground truth masks
        bce_loss = self.bce(inputs, targets)
        # Convert logits to probabilities for Dice calculation
        probs = torch.sigmoid(inputs)
        # Calculate the intersection as the sum of element-wise multiplication
        intersection = (probs * targets).sum()
        # Compute Dice coefficient; the loss is 1 minus this coefficient
        dice_loss = 1 - (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        # Return the combined loss - key improvement over v1's simple BCE
        return bce_loss + dice_loss

# Training loop with updated PyTorch 2.5.1 API
def train_unet(mat_dir, epochs=96, batch_size=32, lr=1e-3, target_size=(256, 256)):
    # Prepare dataset and dataloader for batch processing
    dataset = MatlabTumorDataset(mat_dir, target_size=target_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Use available GPU or default to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # Instantiate the U-Net model with 64 base filters and transfer to device
    model = UNet().to(device)
    # Initialize the combined Dice+BCE loss function
    criterion = DiceBCELoss()
    # Use Adam optimizer with specified learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Apply cosine annealing scheduler for learning rate decay over epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision training setup uing cuda
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # Enable cuDNN auto-tuning for optimal performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Main training loop over epochs
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in loader:
            # Transfer batch to GPU if available
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()  # Clear gradients

            if torch.cuda.is_available() and scaler is not None:
                # Mixed precision training setup using cuda
                with torch.amp.autocast('cuda'):
                    outputs = model(images)  # Forward pass: output logits
                    loss = criterion(outputs, masks)  # Compute combined Dice+BCE loss

                # Backpropagation with scaled loss to avoid underflow in float16
                scaler.scale(loss).backward()
                # Step optimizer: unscale and perform parameter update
                scaler.step(optimizer)
                # Update the scale factor for next iteration
                scaler.update()
            else:
                # CPU fallback without mixed precision
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            # Accumulate loss for monitoring
            running_loss += loss.item()

        # Update learning rate according to schedule
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(loader):.4f}")

    # Save trained model weights
    torch.save(model.state_dict(), "unet_brain_tumor_v1.3.pth")
    print("NOSAv2 model saved as unet_brain_tumor_v1.3.pth")

if __name__ == "__main__":
    train_unet(mat_dir="C:/Matura/data", epochs=96, batch_size=32, lr=1e-3, target_size=(256, 256))
