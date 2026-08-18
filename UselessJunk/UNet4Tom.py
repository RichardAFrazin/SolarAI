#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:02:47 2024

@author: Richard Frazin
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset


if not torch.cuda.is_available():
   print("Read the Sign.  No GPU, No Service!")
   assert False
device = torch.device('cuda')


"""
    Calculer la taille de la sortie d'une couche de convolution 2D (torch.nn.Conv2d).

    Args:
        input_size (int): La taille de l'entrée (hauteur ou largeur).
        kernel_size (int): La taille du noyau de convolution (int).
        stride (int, optional): Le stride de la convolution (par défaut 1).
        padding (int, optional): Le padding ajouté autour de l'image (par défaut 0).

    Returns:
        int: La taille de la sortie après la convolution.
"""

def conv2d_output_size(input_size, kernel_size, stride=1, padding=0):
    output_size = 1 + (input_size - kernel_size + 2 * padding) // stride
    return output_size

"""
Calculate the output size of a ConvTranspose2d layer.

Args:
    input_size (int): The size of the input (height or width).
    kernel_size (int): The size of the kernel (int).
    stride (int, optional): The stride of the transposed convolution (default 1).
    padding (int, optional): The padding added to the input (default 0).
    output_padding (int, optional): Additional padding added to the output (default 0).

Returns:
    int: The size of the output after the transposed convolution.
"""
def ConvTranspose2D_output_size(input_size, kernel_size, stride=1, padding=0, output_padding=0):
    output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
    return output_size

#this assumes images with a single input channel
#The sizes and number of channels are designed for 64x64 input images

class UNetWithSkip(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        # Encoder
        self.enc1 = self.conv_block(1, 4, kernel_size=5, stride=2, padding=2)  # 64x64 -> 32x32
        self.enc2 = self.conv_block(4, 8, kernel_size=5, stride=2, padding=2)  # 32x32 -> 16x16
        self.enc3 = self.conv_block(8, 16, kernel_size=3, stride=2, padding=1)  # 16x16 -> 8x8 (Bottleneck)

        # Decoder
        self.upconv3 = self.upconv_block(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)  # 8x8 -> 16x16
        self.upconv2 = self.upconv_block(16, 4, kernel_size=5, stride=2, padding=2, output_padding=1)  # 16x16 -> 32x32
        self.upconv1 = self.upconv_block(8, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)  # 32x32 -> 64x64

        # Output layer
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def upconv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.ReLU(inplace=True)
        )

    def conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)  # 64x64 -> 32x32
        enc2_out = self.enc2(enc1_out)  # 32x32 -> 16x16
        enc3_out = self.enc3(enc2_out)  # 16x16 -> 8x8 (Bottleneck)

        # Decoder
        up3_out = self.upconv3(enc3_out)  # 8x8 -> 16x16
        up2_out = self.upconv2(torch.cat([up3_out, enc2_out], 1))  # 16x16 + 16x16 -> 32x32
        up1_out = self.upconv1(torch.cat([up2_out, enc1_out], 1))  # 32x32 + 32x32 -> 64x64

        # Output
        out = self.out_conv(up1_out)  # 64x64 -> out_channels
        return out



class ImageDataSet(torch.utils.data.Dataset):
    def __init__(self, input_images, target_images, transform=None):
        """
        Args:
            input_images (list of np.array): List of input images, each of shape (H, W)
            target_images (list of np.array): List of target images, each of shape (H, W)
            transform (callable, optional): Optional transform to be applied to the input and target.
        """
        self.input_images = input_images  # List of input images as np.array
        self.target_images = target_images  # List of target images as np.array
        self.transform = transform  # Optional transform

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.input_images)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            input_img (tensor): A tensor of shape (1, H, W), with one channel
            target_img (tensor): Same as input_img but for the target image.
        """
        input_img = self.input_images[idx]  # Get the input image (np.array)
        target_img = self.target_images[idx]  # Get the target image (np.array)

        # Convert to tensor and add a channel dimension (1, H, W)
        input_img = torch.tensor(input_img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        target_img = torch.tensor(target_img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # Apply transformation if provided
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img



#  Helper Functions for Saving and Loading Checkpoints and Models
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save the model and optimizer state dict to a file.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch}, loss {loss:.4f} to {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """
    Load the model and optimizer state dict from a checkpoint file.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath}, epoch {epoch}, loss {loss:.4f}")
    return epoch, loss

def save_model_for_inference(model, filepath):
    """
    Save the trained model for inference.
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved for inference to {filepath}")
    return None

def load_model_for_inference(model, filepath):
    """
    Load a trained model for inference.
    """
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded for inference from {filepath}")
    return None

# 5. Training and Checkpoint Saving Logic
def train_model(model, train_loader, optimizer, criterion, epochs, checkpoint_dir, checkpoint_freq):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Save checkpoint at specified frequency
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_filepath = f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, checkpoint_filepath)
    return None



# %%


class UNetNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
       super().__init__()

       # Encoder with modified channels and larger kernels
       self.enc1 = self.conv_block(in_channels, 16, kernel_size=5)
       self.enc2 = self.conv_block(16, 32, kernel_size=5)
       self.enc3 = self.conv_block(32, 64, kernel_size=3)
       self.enc4 = self.conv_block(64, 128, kernel_size=3)

       # Bottleneck
       self.bottleneck = self.conv_block(128, 64, kernel_size=3)

       # Decoder with reduced channels (no skip connections)
       self.dec3 = self.conv_block(64, 64, kernel_size=3)  # Pas de concaténation
       self.dec2 = self.conv_block(64, 32, kernel_size=3)
       self.dec1 = self.conv_block(32, 16, kernel_size=3)

       # Output layer
       self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, kernel_size=3):
       """Basic convolution block with optional kernel size adjustment"""
       return nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
           nn.ReLU(inplace=True),
           nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
           nn.ReLU(inplace=True)
       )

    def forward(self, x):
       debug = False
       if debug:
           print(f"Input shape: {x.shape}")

       # Encoder
       enc1 = self.enc1(x)
       if debug:
           print(f"After enc1: {enc1.shape}")
       enc2 = self.enc2(F.max_pool2d(enc1, 2))
       if debug:
           print(f"After enc2: {enc2.shape}")
       enc3 = self.enc3(F.max_pool2d(enc2, 2))
       if debug:
           print(f"After enc3: {enc3.shape}")
       enc4 = self.enc4(F.max_pool2d(enc3, 2))
       if debug:
           print(f"After enc4: {enc4.shape}")

       # Bottleneck
       bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
       if debug:
           print(f"After bottleneck: {bottleneck.shape}")

       # Decoder (sans skip connections)
       dec3 = self.dec3(F.interpolate(bottleneck, size=(7,7), mode='bilinear', align_corners=False))
       if debug:
           print(f"After dec3 (upscaled): {dec3.shape}")
       dec2 = self.dec2(F.interpolate(dec3, size=(15,15), mode='bilinear', align_corners=False))
       if debug:
           print(f"After dec2 (upscaled): {dec2.shape}")
       dec1 = self.dec1(F.interpolate(dec2, size=(31,31), mode='bilinear', align_corners=False))
       if debug:
           print(f"After dec1 (upscaled): {dec1.shape}")

       # Output
       out = self.out_conv(dec1)
       if debug:
           print(f"Output shape: {out.shape}")

       return out
