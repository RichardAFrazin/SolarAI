#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:58:37 2026

@author: rfrazin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

#This UNet takes set of backprojection images (each at perhaps a different time) as input,
#   and it creates an output tensor of 2D images at various times.
#   Each backprojection is an 80x80 and found from applying the adjoint (transpose) of the projection
#   operator for angle k to y_k, which is the projection obtained at angle k.
#   Thus, the input size is (n_projs, 80, 80)
#        the  output size is (n_output_times, 80,80)
# n_input_chan - number of input channels.  This is at least the number of
#    backprojections, which are likely to be at different times.  Allowing n_input_chan
#    to be > than the number of backprojections allows additional input images, such
#    as regularized least-square solutions.
#n_output_times - the number of time points in the output reconstruction.  This is
#    the number of output channels (each channel is an image).

class TomoUNet(nn.Module):
    def __init__(self, n_input_chan, n_output_times, n_input_times):  # don't need to include batch dimension
        super().__init__()

        if n_input_times > n_input_chan:
           raise ValueError("n_input_times must be <= n_input_chan")
        self.n_input_times = n_input_times
        self.time_ker_size = 25

        self.time_network = nn.Sequential( # 1D dense network for input times
             nn.Linear(n_input_times, 50),
             nn.ReLU(inplace=True),
             nn.Linear(50, 50),
             nn.ReLU(inplace=True),
             nn.Linear(50, self.time_ker_size)     )

        # 1. fonctions de l'ENCODEUR (Descente)
        # Reçoit (n_input_chan, 80, 80) -> Sort (64, 80, 80)
        self.inc = DoubleConv(n_input_chan, 64)
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

        """
         Calcule le noyau de convolution 1D dynamique à partir des temps d'observation.
         input_times : tenseur 1D de taille (N,)
         Retourne : un vecteur s de taille (25,)
        """


    def forward(self, x, input_times=None):  #  need to include batch dimension
        if input_times is None:
           raise ValueError("input_times in form of torch.tensor on the correct device must be supplied.")

        def TimeKernel1D(times_of_input):
          if len(times_of_input[0,:]) != self.n_input_times:  # it must have a batch dimension
             raise ValueError("len(input_times) must equal n_input_times.")
          n_batch = times_of_input.shape[0]
          timekers = torch.zeros(n_batch, self.time_ker_size, device=times_of_input.device)
          for k in range(n_batch):
                 s = self.time_network(times_of_input[k,:].unsqueeze(0))  # add batch dimension
                 timekers[k,:] = s.squeeze(0)
          return timekers

        # ─── ENCODEUR ───
        x1 = self.inc(x)           # (64, 80, 80) -> Sauvegardé pour Skip Connection 1
        p1 = self.pool1(x1)        # (64, 40, 40)
        x2 = self.down1(p1)        # (28, 40, 40) -> Sauvegardé pour Skip Connection 2
        p2 = self.pool2(x2)        # (128, 20, 20)

        # ─── GOULOT D'ÉTRANGLEMENT ───
        b = self.bottleneck(p2)    # (256, 20, 20)
        n_batch, channels, height, width = b.shape
        pad_size = self.time_ker_size//2
        b_conv_total = torch.zeros_like(b)

        time_kers = TimeKernel1D(input_times)  # output (batch, self.time_ker_size)

        for k in range(n_batch): # F.conv1D only does the convolution along axis=2
            b_sample = b[k].view(channels, height*width)
            b_padded = torch.cat([b_sample[-pad_size:, :], b_sample, b_sample[:pad_size, :]], dim=0) # (280, 400)
            b_padded = b_padded.permute(1, 0).unsqueeze(1) # output (400, 1, 280)
            time_ker = time_kers[k].view(1,1, self.time_ker_size)
            res_conv = F.conv1d(b_padded, time_ker, padding=0)  # output (400,1,256)
            res_conv = res_conv.squeeze(1).permute(1,0) # output (256,400)
            res_conv = res_conv.view(channels, height, width)  # output (256,20,20)
            b_conv_total[k] = res_conv

        # ─── DÉCODEUR ───
        u1 = self.up1(b_conv_total)           # (B, 256, 40, 40)
        c1 = torch.cat([u1, x2], dim=1) # Concaténation le long de l'axe des canaux -> (B, 384, 40, 40)
        d1 = self.up_conv1(c1)  # d1.shape = (B, 128, 40, 40)

        u2 = self.up2(d1)  # u2.shape = (B, 128,80,80)
        c2 = torch.cat([u2,x1] ,dim=1)  # c2.shape = (B, 192, 80, 80)
        d2 = self.up_conv2(c2)     # (B, 64, 80, 80)
        out = self.out_conv(d2)    # (B, n_output_times, 80, 80)
        out = self.activation_positive(out)

        return out

# End of class TomoUNet

# --- SCRIPT DE TEST RAPIDE ---
if __name__ == "__main__":
    # Paramètres de votre problème-jouet
    n_input_chan=11; n_output_times = 15; n_input_times = 9

    # Instance du modèle
    model = TomoUNet(n_input_chan, n_output_times, n_input_times)

    # Simulation d'un lot (batch) de 4 exemples de données d'entrée
    donnees_test = torch.rand(4, n_input_chan, 80, 80)  # batch of 4
    times = torch.rand(4, n_input_times)

    # Passage dans le modèle
    prediction = model(donnees_test, times)

    print("Forme de l'entrée (Projections) :", donnees_test.shape)
    print("Forme de la sortie (Images)     :", prediction.shape)
