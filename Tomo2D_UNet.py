#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:58:37 2026

@author: rfrazin
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module): # this does not alter the 80x80 image size, but the number of channels changes
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

#This UNet takes set of backprojection image (each at perhaps a different time) as input,
#   and it creates an output tensor of 2D images at various times.
#   Each backprojection is an 80x80 and found from applying the adjoint (transpose) of the projection
#   operator for angle k to y_k, which is the projection obtained at angle k.
#   Thus, the input size is (n_projs, 80, 80)
#        the  output size is (n_output_times, 80,80)
#n_projs - number of input projections (perhaps a different times)
#n_output_times - the number of time points in the output reconstruction

class TomoUNet(nn.Module):
    def __init__(self, n_input_angles, n_output_times):
        super().__init__()

        # 1. fonctions de l'ENCODEUR (Descente)
        # Reçoit (n_input_angles, 80, 80) -> Sort (64, 80, 80)
        self.inc = DoubleConv(n_input_angles, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 80x80 -> 40x40

        # Reçoit (64, 40, 40) -> Sort (128, 40, 40)
        self.down1 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 40x40 -> 20x20

        # 2. GOULOT D'ÉTRANGLEMENT (Bottleneck)
        # Reçoit (128, 20, 20) -> Sort (256, 20, 20)
        self.bottleneck = DoubleConv(128, 256)

        # 3. fonctions du DÉCODEUR (Remontée avec interpolation)
        #nn.Upsample double la taille spatiale (20x20 -> 40x40) sans changer les canaux
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Après concaténation avec l'encodeur, on a 128 + 128 = 256 canaux en entrée
        self.up_conv1 = DoubleConv(384, 128)

        # Double la taille spatiale (40x40 -> 80x80)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Après concaténation avec le tout début, on a 64 + 64 = 128 canaux en entrée
        self.up_conv2 = DoubleConv(192, 64)

        # 4. COUCHE DE SORTIE (Changement de dimension : Canaux -> Temps)
        # Une convolution 1x1 suffit pour projeter les 64 filtres vers le nombre d'instants voulus
        self.out_conv = nn.Conv2d(64, n_output_times, kernel_size=1)

        self.activation_positive = nn.Softplus() # Guarantee positivity.  Softplus(x) = log(1 + exp(x))

    def forward(self, x):
        # ─── ENCODEUR ───
        x1 = self.inc(x)           # (Batch, 64, 80, 80) -> Sauvegardé pour Skip Connection 1
        p1 = self.pool1(x1)        # (Batch, 64, 40, 40)

        x2 = self.down1(p1)        # (Batch, 128, 40, 40) -> Sauvegardé pour Skip Connection 2
        p2 = self.pool2(x2)        # (Batch, 128, 20, 20)

        # ─── GOULOT D'ÉTRANGLEMENT ───
        b = self.bottleneck(p2)    # (Batch, 256, 20, 20)

        # ─── DÉCODEUR ───
        u1 = self.up1(b)           # (Batch, 256, 40, 40)
        c1 = torch.cat([u1, x2], dim=1) # Concaténation le long de l'axe des canaux -> (Batch, 384, 40, 40)
        d1 = self.up_conv1(c1)  # d1.shape = (Batch, 128, 40, 40)

        u2 = self.up2(d1)  # u2.shape = (Batch, 128,80,80)
        c2 = torch.cat([u2,x1] ,dim=1)  # c2.shape = (Batch, 192, 80, 80)
        d2 = self.up_conv2(c2)     # (B, 64, 80, 80)
        out = self.out_conv(d2)    # (B, n_output_times, 80, 80)
        out = self.activation_positive(out)

        return out



# End of class TomoUNet

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
