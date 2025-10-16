import torch
import torch.nn as nn

# Double convolution block with dropout for regularization
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
        x4_resized = x4         # Skip connection from encoder block 4
        x = self.conv1(torch.cat([x, x4_resized], dim=1))  # 1024 channels -> 256 filters
        x = self.up2(x)         # upsampled to 256 filters
        x3_resized = x3         # Skip connection from encoder block 3
        x = self.conv2(torch.cat([x, x3_resized], dim=1))  # 512 channels -> 128 filters
        x = self.up3(x)         # upsampled to 128 filters
        x2_resized = x2         # Skip connection from encoder block 2
        x = self.conv3(torch.cat([x, x2_resized], dim=1))  # 256 channels -> 64 filters
        x = self.up4(x)         # upsampled to 64 filters
        x1_resized = x1         # Skip connection from encoder block 1
        x = self.conv4(torch.cat([x, x1_resized], dim=1))  # 128 channels -> 64 filters

        x = self.outc(x)
        return x  # NOSAv2: Returns raw logits (no sigmoid)

# define the model with parameters
model = UNet(n_channels=1, n_classes=1, base_filters=64)
