#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:58:37 2026

@author: rfrazin
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Bloc de base : deux convolutions successives avec Batch Normalization et ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)        )
    def forward(self, x):
        return self.conv(x)

#This UNet takes set of 1D projections (each at perhaps a different time) as input,
#   and it creates an output tensor of 2D images at various times.
#n_projs - number of input projections (perhaps a different times)
#n_rays - number of rays (data points) per projection
#n_output_times - the number of time points in the output reconstruction
class TomoUNet(nn.Module):
    def __init__(self, n_projs, n_rays, n_output_times):
        super().__init__()
        self.M = n_projs
        self.n_rays = n_rays
        self.n_output_times = n_output_times

        # 1. Adaptateur de dimension : (M * 130) -> ((N + 1) * 80 * 80)
        in_features = n_projs*n_rays
        out_features = n_output_times*80*80
        self.adaptor = nn.Linear(in_features, out_features)

        # 2. Composants du U-Net
        # Encodeur (Descente)
        self.inc = DoubleConv(n_output_times, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))       # 80x80 -> 40x40
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))      # 40x40 -> 20x20
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))      # 20x20 -> 10x10
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))     # 10x10 -> 5x5 (Bottleneck)

        # Décodeur (Remontée) avec couches d'up-sampling
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)      # 5x5 -> 10x10
        self.conv_up1 = DoubleConv(1024, 512)                                  # 512 (up) + 512 (skip)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)       # 10x10 -> 20x20
        self.conv_up2 = DoubleConv(512, 256)                                   # 256 (up) + 256 (skip)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)       # 20x20 -> 40x40
        self.conv_up3 = DoubleConv(256, 128)                                   # 128 (up) + 128 (skip)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)        # 40x40 -> 80x80
        self.conv_up4 = DoubleConv(128, 64)                                    # 64 (up) + 64 (skip)

        # Couche de sortie finale
        self.outc = nn.Conv2d(64, n_output_times, kernel_size=1)

    def forward(self, x):
        # x a une forme initiale de (Batch_Size, M, 130)
        batch_size = x.size(0)

        # Étape 1 : Aplatir et passer par l'adaptateur linéaire
        x = x.view(batch_size, -1)
        x = self.adaptor(x)

        # Redimensionner au format image (Batch_Size, Canaux_Temporels, H, W)
        x = x.view(batch_size, self.N_plus_1, 80, 80)

        # Étape 2 : Boucle U-Net avec connexions transversales (Skip Connections)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # Bottleneck (5x5 pixels)

        # Remontée avec concaténation des caractéristiques de l'encodeur
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        logits = self.outc(x)
        return logits

# --- SCRIPT DE TEST RAPIDE ---
if __name__ == "__main__":
    # Paramètres de votre problème-jouet
    M_angles = 20
    detectors = 130
    N_instants = 30 # N+1 vaudra donc 31 canaux

    # Instance du modèle
    model = TomoUNet(M=M_angles, n_rays=detectors, N_plus_1=N_instants + 1)

    # Simulation d'un lot (batch) de 4 exemples de données d'entrée
    donnees_test = torch.rand(4, M_angles, detectors)

    # Passage dans le modèle
    prediction = model(donnees_test)

    print("Forme de l'entrée (Projections) :", donnees_test.shape)
    print("Forme de la sortie (Images)     :", prediction.shape)
