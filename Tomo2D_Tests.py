#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 13:05:24 2026

@author: rfrazin
"""

import numpy as np
from scipy.sparse import csr_matrix as CSR
import matplotlib.pyplot as plt
import Tomo2D_UNet as UN
import ProjectionUtils2D as PU
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm   # training progress bar display



#%%

"""
This creates a training set for the UNet.  The idea is for the UNet to learn to
  extrapolate in time given the backprojection images of the Radon projections.
  The training video ("video") is divided into chunks of "nFramesOut"
  with a stride in integer units specifed by "stride", which can
  be None, in which case the stride is set to nFramesOut//2.
The UNet output has shape ("nFramesOut", 80, 80), which are compared to the corresponding
  truth video images in the loss function, either directly or in projection.
The UNet input shape is ("nAnglesObs", 80,80) and consists of "nAnglesObs" Radon backprojections.
For each training sample, the projections are  acquired over a time interval specified
   by "StartEndObs", which is a list of length 2.  These two numbers are in units
   of frame numbers and must be between 0 and "nFramesOut"-1.

"""
def TrainingDataSet(nFramesOut, nAnglesObs, StartEndObs, stride=None, video=PU.vid1):
   if stride is None: stride = nFramesOut//2
   stride = int(stride)
   if video.ndim != 3:
      raise ValueError(f"video.ndim must be 3, but it has {video.ndim} dimensions.")
   StartEndObs = np.array(StartEndObs)
   if (len(StartEndObs) != 2):
      raise ValueError("The parameter StartEndObs must have length 2.")
   if (StartEndObs[0] < 0.) or (StartEndObs[1] > nFramesOut-1 ) or (StartEndObs[1] < StartEndObs[0]):
      raise ValueError("Invalid values for parameter StartEndObs.")
   if (video.shape[1] != 80) or (video.shape[2] != 80):
      raise ValueError("The video must consist of 80x80 images.")

   #The projection angles are the same for each sample, so only compute them once
   angles = np.linspace(0, np.pi*nAnglesObs/(nAnglesObs-1), nAnglesObs)
   projmats = []
   for angle in angles:
      densemat = PU.ProjectionSubMatrix(angle, setup=PU.setup)  # use sparse format for speed
      projmats.append(CSR(densemat))

   samples = []  # list of input/output tuples

   VideoEnd = False # True signals the end of the video
   k = -1  #sample number
   while not VideoEnd:
      k += 1
      truthvid = video[k*stride : k*stride + nFramesOut, :, :]  # truth video
      bpvid = []  # backprojection video
      times =  np.linspace(StartEndObs[0], StartEndObs[1], nAnglesObs)
      #  get the portion of truthvid corresponding to the observation times
      vidobs = PU.VideoInterpolate(times, truthvid)  # temporal interpolation
      for i in range(vidobs.shape[0]):
         #note the use of parentheses to avoid matrix-matrix multiplication!
         bp = projmats[i].T@(projmats[i]@vidobs[i].reshape((80*80,)))
         bpvid.append( bp.reshape((80,80))  )
      samples.append( (np.array(bpvid), truthvid)  )
      if (k+1)*stride + nFramesOut >= video.shape[0]:  #terminate while loop
         VideoEnd=True

   return samples

#%%

# 1. Dataset pour les paires (observation, vérité)
class TomoDataset(Dataset):
    def __init__(self, samples_list):
        self.samples = samples_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        bp_img, truth_img = self.samples[idx]
        # Conversion obligatoire en tenseurs PyTorch float32
        x = torch.from_numpy(bp_img).float()
        y = torch.from_numpy(truth_img).float()
        return x, y

# 2. Fonction pour entraîner le modèle sur une époque
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    # Barre de progression pour suivre l'avancement
    progress_bar = tqdm(dataloader, desc="Entraînement", leave=False)

    for bkprojs, targets in progress_bar:
        bkprojs = bkprojs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Passage avant (Forward pass)
        outputs = model(bkprojs)   # Sortie attendue: (B, n_output_times, 80, 80)

        # loss contains the current loss  value (scalar) and a calculation graph
        loss = criterion(outputs, targets)
        loss.backward()  # gradient calculation via backprop
        optimizer.step()

        running_loss += loss.item() * bkprojs.size(0)
        progress_bar.set_postfix({'loss': loss.item()})

    return running_loss / len(dataloader.dataset)

def validate_one_epoch(model, dataloader_val, criterion, device):
    # 1. Activation du mode évaluation
    model.eval()
    running_loss = 0.0

    # 2. Désactivation du calcul des gradients
    with torch.no_grad():
        for observations, truth in dataloader_val:
            observations = observations.to(device)
            truth = truth.to(device)
            # Passage avant uniquement
            reconstructions = model(observations)
            # Calcul du coût de validation
            loss = criterion(reconstructions, truth)
            running_loss += loss.item() * observations.size(0)
    return running_loss / len(dataloader_val.dataset)


#%% 3. Exemple de script principal pour lancer l'entraînement
if __name__ == "__main__":
    # Détection du GPU (CUDA / MPS pour Mac) ou CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        kbd = input("cuda device not found.  Resorting to CPU. Press 'c' to continue.")
        if kbd.lower() != "c" : exit()
        else: pass
    print(f"Using {device} device for training.")

    # Initialization.  498 samples with TrainingDataSet(16,25,[3.,13.],stride=6)
    samp = TrainingDataSet(16,25,[3.,13.],stride=6)  #list of samples
    samp_train = samp[:450]
    samp_validation = samp[450:]
    dataset_train = TomoDataset(samp_train)
    dataset_val = TomoDataset(samp_validation)
    dataloader_train = DataLoader(dataset_train, batch_size=10, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=10, shuffle=False)

    # Initialisation du modèle (25 angles d'entrée, 16 instants de sortie)
    model = UN.TomoUNet(n_input_angles=25, n_output_times=16).to(device)

    # Fonction de perte (MSE) et Optimiseur (Adam)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#%%
   # Boucle sur plusieurs époques
    num_epochs = 100
    for epoch in range(num_epochs):
        loss_train = train_one_epoch(model, dataloader_train, criterion, optimizer, device)
        loss_val = validate_one_epoch(model, dataloader_val, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] -> "
              f"Training loss : {loss_train:.6f} | "
              f"Validation : {loss_val:.6f}")

#%% Look at results from the validation set.  See TrainingDataSet function to see how samples are created.
# sdex - sample index
# model - the UNet model
# samplist - list of samples, each element of which is an (observation, target) tuple.

def ViewResults(sdex, model, samplist):
   obs   = samplist[sdex][0]
   targ   = samplist[sdex][1]
   n_ang = len(obs)
   angs = np.linspace(0, 2*np.pi*(n_ang-1)/n_ang, n_ang)




   return None


snum = 42
truset = samp_validation[snum][1]  # truth images
bprset = torch.from_numpy(samp_validation[snum][0]).float().unsqueeze(0).to(device)  # input back projections
recset = (model(bprset).detach().cpu().numpy()).squeeze(axis=0)

imdx = [0, 3, 6, 9, 12, 15]
for dx in imdx:
   im = np.vstack((truset[dx,:,:],recset[dx,:,:]))
   plt.figure(dx)
   plt.imshow(im, cmap='coolwarm');plt.colorbar()
