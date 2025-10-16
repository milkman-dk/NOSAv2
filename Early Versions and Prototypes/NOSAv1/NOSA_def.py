import torch
import numpy as np
import mat73
import os

# ---- 1. Define the UNet model ----
class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super(DoubleConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(torch.nn.Module):
    def __init__(self, n_channels=1, n_classes=1, base_filters=32):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(base_filters, base_filters*2))
        self.down2 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(base_filters*2, base_filters*4))
        self.down3 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(base_filters*4, base_filters*8))
        self.down4 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(base_filters*8, base_filters*8))
        self.up1 = torch.nn.ConvTranspose2d(base_filters*8, base_filters*8, 2, stride=2)
        self.conv1 = DoubleConv(base_filters*8 + base_filters*8, base_filters*4)
        self.up2 = torch.nn.ConvTranspose2d(base_filters*4, base_filters*4, 2, stride=2)
        self.conv2 = DoubleConv(base_filters*4 + base_filters*4, base_filters*2)
        self.up3 = torch.nn.ConvTranspose2d(base_filters*2, base_filters*2, 2, stride=2)
        self.conv3 = DoubleConv(base_filters*2 + base_filters*2, base_filters)
        self.up4 = torch.nn.ConvTranspose2d(base_filters, base_filters, 2, stride=2)
        self.conv4 = DoubleConv(base_filters + base_filters, base_filters)
        self.outc = torch.nn.Conv2d(base_filters, n_classes, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.conv1(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x, x2], dim=1))
        x = self.up4(x)
        x = self.conv4(torch.cat([x, x1], dim=1))
        x = self.outc(x)
        return self.sigmoid(x)
