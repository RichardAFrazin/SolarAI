#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 13:05:24 2026

@author: rfrazin
"""

import os
import warnings
import numpy as np
from scipy.sparse import csr_matrix as CSR
import matplotlib.pyplot as plt
import Tomo2D_UNet as UN
import ProjectionUtils2D as PU
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm   # training progress bar display
import glob
import imageio.v2 as imageio



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
If RandomProjTimes is True, the acquistion times are randomly distributed between StartEndObs[0]
   and StartEndObs[1].  If not, they are evenly distributed.
If UseSolLS is True, a static least-squares solution (x_ls) is computed from the projections and
   included as an additional input channel
regparam - only matters if UseSolLS is True.  Can have more than one value --> more channels added

"""
def TrainingDataSet(nFramesOut, nAnglesObs, StartEndObs, stride=None, RandomProjTimes=False, UseSolLS=False, regparam=[0.1,0.01], video=PU.vid1):
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

   def RandomTimes(n, StartEnd):  #Returns random obs. times and corresponding angles
      dt = np.random.rand(n)
      dt /= np.sum(dt)
      angles = np.pi*np.cumsum(dt)
      dt = (StartEnd[1]-StartEnd[0])*np.cumsum(dt)
      times = StartEnd[0] + dt
      return   (times, angles)

   ProjTimes  = []
   ProjAngles = []
   samples = []  # list of input/output tuples
   if UseSolLS:
      regparam = list(regparam) # make it iterable
      SolsLS = []  # put the regularized least-squares solutions here
   else:
      SolsLS = None

   if RandomProjTimes is False:  # default is evenly distributed data acquisition times and angles
      times =  np.linspace(StartEndObs[0], StartEndObs[1], nAnglesObs)
      angles = np.linspace(0, np.pi*nAnglesObs/(nAnglesObs-1), nAnglesObs)
      ProjMats = []
      for angle in angles:
          ProjMats.append(CSR(PU.ProjectionSubMatrix(angle, setup=PU.setup)))


   # Loop over the video
   VideoEnd = False # True signals the end of the video
   k = -1  #sample number
   while not VideoEnd:
      k += 1   # k is the sample index
      truthvid = video[k*stride : k*stride + nFramesOut, :, :]  # truth video
      bpvid = []  # backprojection video

      if RandomProjTimes:
         (times, angles) = RandomTimes(nAnglesObs, StartEndObs)
         ProjMats = []
         for angle in angles:
            ProjMats.append(CSR(PU.ProjectionSubMatrix(angle, setup=PU.setup)))

      ProjTimes.append( times) # appended whether or not RandomProjTimes is True
      ProjAngles.append(angles) # appended whether or not RandomProjTimes is True

      #  get the video interpolated to the observation times
      vidobs = PU.VideoInterpolate(times, truthvid)  # temporal interpolation
      for i in range(len(angles)):
         proj_i = ProjMats[i]@vidobs[i].reshape((80*80,))  # projection
         bp_i = ProjMats[i].T@proj_i  # backprojection
         bpvid.append( bp_i.reshape((80,80)) )  # bpvid is set of images that are input to the UNet.
      if UseSolLS:  # get regularized dleast-squares solutions.  These solutions seems to help the UNet when they supplement the backprojections
         solsk = []
         with warnings.catch_warnings():
               warnings.simplefilter("ignore",category=UserWarning)
               for reg in regparam:
                   x_ls = PU.StaticReconstruction(truthvid, ProjMats, times, RegFcn='Nabla_sparse', regparam=reg, UseTorch=True, ShowSolver=False)
                   bpvid.append( x_ls.reshape((80,80)) )  # unlike above, this is NOT a backprojection.
                   solsk.append( x_ls.reshape((80,80)) )
               SolsLS.append(solsk)

      samples.append( (np.array(bpvid), truthvid)  )
      if (k+1)*stride + nFramesOut >= video.shape[0]:  #terminate while loop
         VideoEnd=True

   ReturnD = {'samples' : samples, 'ProjTimes': ProjTimes, 'ProjAngles': ProjAngles, 'SolsLS': SolsLS}

   return ReturnD

#%%

# 1. Dataset pour les paires (observation, vérité)
class TomoDataset(Dataset):
    def __init__(self, samples, proj_times): # data_dict):
        self.samples = samples  # self.samples = data_dict['samples']
        self.proj_times = proj_times # self.proj_times = data_dict['ProjTimes']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        bp_img, truth_img = self.samples[idx]
        times = self.proj_times[idx]
        # Conversion obligatoire en tenseurs PyTorch float32
        x = torch.from_numpy(bp_img).float()
        y = torch.from_numpy(truth_img).float()
        input_times = torch.from_numpy(times).float()
        return x, y, input_times

# 2. Fonction pour entraîner le modèle sur une époque
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    # Barre de progression pour suivre l'avancement
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for bkprojs, targets, input_times in progress_bar:
        bkprojs = bkprojs.to(device)
        targets = targets.to(device)
        input_times = input_times.to(device)

        optimizer.zero_grad()

        # Passage avant (Forward pass)
        outputs = model(bkprojs, input_times)   # Sortie attendue: (B, n_output_times, 80, 80)

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
        for observations, truth, input_times in dataloader_val:
            observations = observations.to(device)
            truth = truth.to(device)
            input_times = input_times.to(device)

            # forward pass through network
            reconstructions = model(observations, input_times)
            # Calcul du coût de validation
            loss = criterion(reconstructions, truth)
            running_loss += loss.item() * observations.size(0)
    return running_loss / len(dataloader_val.dataset)


def TrainModel(model, dataloader_train, dataloader_val,
               criterion, optimizer, device, num_epochs=50):
    for epoch in range(num_epochs):
        loss_train = train_one_epoch(model, dataloader_train, criterion, optimizer, device)
        loss_val = validate_one_epoch(model, dataloader_val, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] -> "
              f"Training loss : {loss_train:.6f} | "
              f"Validation : {loss_val:.6f}")
    return model

# fnameWpath is the filename with the path included.  Recommended suffix: .pt
def SaveModelWeights(model, fnameWpath):
    torch.save(model.state_dict(), fnameWpath)
    return None
def LoadModelWeights(model, fnameWpath):
   return model.load_state_dict(torch.load(fnameWpath, weights_only=True))
def SaveCheckpoint(model, optimizer, epoch, loss, fnameWpath):
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,}, fnameWpath)
    return None
def LoadCheckpoint(model, optimizer, fnameWpath):
     checkpoint = torch.load(fnameWpath, weights_only=True)
     model.load_state_dict(checkpoint['model_state_dict'])
     model.eval()  # safe evaluation practice
     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
     epoch = checkpoint['epoch']
     loss = checkpoint['loss']
     return (model,optimizer, epoch, loss)

#%% Look at results from the validation set.  See TrainingDataSet function to see how samples are created.
# This also carries out a static reconstruction in order to compare to the dynamic results
# sdex - sample index
# model - the UNet model
# TDSoutput - dict output of TrainingDataSet.
# regparam - regularization paramter for the static reconstruction
# ReturnRMS returns RMS errors if True
def ViewResults(sdex, model, TDSoutput, regparam=0.1, ReturnRMS= False, device="cuda"):
   samplist = TDSoutput['samples']
   obs   = samplist[sdex][0]  # observations (UNet input)
   targ   = samplist[sdex][1] # target video frames (UNet output)
   times  = TDSoutput['ProjTimes'][sdex]
   angles = TDSoutput['ProjAngles'][sdex]

   ProjMats = []
   for ang in angles:
      ProjMats.append( PU.ProjectionSubMatrix(ang, setup=PU.setup) )

   x_static = PU.StaticReconstruction(targ, ProjMats, times, RegFcn='Nabla_sparse', regparam=regparam, ShowSolver=False)
   x_true = PU.VideoInterpolate(np.median(times), targ)

   model.eval()  #disable dropout (enabled by default) for deterministic evaluation
   with torch.no_grad():  # reconstructed video
      vid_UNet = model( torch.from_numpy(obs).float().unsqueeze(0).to(device) )
      vid_UNet = (vid_UNet.detach().cpu().numpy()).squeeze(axis=0)
   x_UNet = PU.VideoInterpolate(np.median(times), vid_UNet)

   image_out = np.hstack((x_true, x_UNet, x_static));
   rms = [np.std(x_UNet-x_true), np.std(x_static - x_true)]
   print(f"RMS errors: UNet {rms[0]}, static {rms[1]}.")
   plt.figure();
   im = plt.imshow(image_out, cmap='coolwarm');
   plt.colorbar(im, fraction=0.02, pad=0.04);
   for text, px_pos in zip(["True", "UNet", "Reg. LstSqu"], [40, 120, 200]):
       plt.text(px_pos, -5, text, fontsize=10, ha='center', weight='bold')

   if ReturnRMS:
      return (image_out, rms)
   else:
      return image_out

#%%
#
def SaveViewResultsOnDisk(model, TDSoutput, StartEndObs, output_dir,
                          ProjMats=None, device="cuda"):
    print("Warning: This closes all figures!")

    samplist = TDSoutput['samples']

    os.makedirs(output_dir, exist_ok=True)
    plt.ioff() # Évite d'ouvrir des fenêtres pop-up qui saturent l'écran

    print(f"Saving the frames to '{output_dir}'...")

    # Dimensions exactes pour obtenir 900 x 350 pixels à 150 DPI
    target_width_inches = 900 / 150  # 6.0
    target_height_inches = 350 / 150 # 2.333333...

    for sdex in range(len(samplist)):
        fig = plt.figure(figsize=(target_width_inches, target_height_inches))
        # Optionnel mais recommandé : configurer une géométrie d'axes fixe dans la figure
        # [gauche, bas, largeur, hauteur] en fractions de la figure (de 0 à 1)
        # Cela laisse 15% de marge à gauche/bas pour les légendes sans changer la taille globale
        ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

        _ = ViewResults(sdex, model, TDSoutput, regparam=0.1, ReturnRMS= False, device="cuda")

        # Sauvegarde au format PNG ou JPG avec un index fixe à 3 chiffres (000, 001, 002...)
        plt.savefig(f"{output_dir}/frame_{sdex:03d}.png", dpi=150)
        plt.close('all') # Nettoie la mémoire RAM immédiatement
    plt.ion()
    return None

#  input_dir - location of input images
def CompileImagesToVideo(input_dir, gif_output_filename, fps=3):
    files = sorted(glob.glob(f"{input_dir}/frame_*.png"))  #  glob.glob enables wildcard characters in a directory search
    if not files:
        print(f"Not matching files found in {input_dir}")
        return

    # 2. Lire les images sur le disque
    images = [imageio.imread(f) for f in files]

    # 3. Sauvegarder le fichier final (.gif ou .mp4 selon l'extension choisie)
    print(f"Création du fichier {gif_output_filename}...")
    # imageio gère automatiquement le format .gif ou .mp4 selon le nom
    imageio.mimsave(gif_output_filename, images, fps=fps)
    print("Video created and saved.")

#%% 3. training script
if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        kbd = input("cuda device not found.  Resorting to CPU. Press 'c' to continue.")
        if kbd.lower() != "c" : exit()
        else: pass
    print(f"Using {device} device.")

    n_input_angles = 25

    inp1 = input("If you want to build the UNet and train it, type 'B'. If you want to work with the trained model to make images and videos, type 'O'.")
    inp1 = inp1.upper()
    if inp1 not in ['O','B']: raise ValueError("Must choose 'O' or 'B'.")
    if inp1 == 'B':
       inp2 = input("Do you want to include regularized static least-squares solutions as an inputs?  [Y]es or [N]o.")
       if inp2.upper() == 'N' : UseSolLS = False
       elif inp2.upper() == 'Y': UseSolLS = True
       else: raise ValueError("Must choose 'Y' or 'N'.")
    inp3 = input("Do you want random observation times within the interval specified by StartEndObs?  [Y] or [N]o.")
    if inp3.upper() == 'N'  : RandomProjTimes = False
    elif inp3.upper() == 'Y': RandomProjTimes = True
    else: raise ValueError("Must choose 'Y' or 'N'.")

    output_dir = "./video_frames4"
    videoname = "UNet_rndTime_2LS_results.gif"
#%%
    if inp1 == 'B':

       # Initialization.  498 samples with TrainingDataSet nFramesOut=16, StartEndObs=[3.,13.] ,stride=6

       out = TrainingDataSet(16, n_input_angles, [3.,13.],stride=6, RandomProjTimes=RandomProjTimes,
                             UseSolLS=UseSolLS, regparam=[0.1,0.01],video=PU.vid1)

       samp = out['samples']
       times = out['ProjTimes']
       if UseSolLS:
          sol = out['SolsLS']  # LS solutions

       n_input_chan  = samp[0][0].shape[0]  # may not equal n_input_times
       n_output_chan = samp[0][1].shape[0]  # number of output times
       n_input_times = len(times[0])
       samp_train = samp[:450]
       times_train = times[:450]
       samp_validation = samp[450:]
       times_val = times[450:]
       dataset_train = TomoDataset(samp_train, times_train)
       dataset_val = TomoDataset(samp_validation, times_val)
       dataloader_train = DataLoader(dataset_train, batch_size=10, shuffle=True)
       dataloader_val = DataLoader(dataset_val, batch_size=10, shuffle=False)
       model = UN.TomoUNet(n_input_chan, n_output_chan, n_input_times).to(device)
       criterion = nn.MSELoss()
       optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
       model = TrainModel(model, dataloader_train, dataloader_val,
                          criterion, optimizer, device, num_epochs=50)
    elif inp1 == 'O':
      print("This assumes the UNet is in memory as 'model', among other things.")
      nAnglesObs = 25; StartEndObs  = [3.,13.]
      out_fine = TrainingDataSet(16, nAnglesObs, StartEndObs, stride=1, RandomProjTimes=RandomProjTimes,
                                 UseSolLS=UseSolLS, video=PU.vid1[2687:])
      angles = np.linspace(0, np.pi*(nAnglesObs-1)/nAnglesObs, nAnglesObs)
      ProjMats = []
      for ang in angles:
         ProjMats.append( CSR(PU.ProjectionSubMatrix(ang, setup=PU.setup)) )
      SaveViewResultsOnDisk(model, out_fine, StartEndObs, output_dir,
                          ProjMats=None, UseSolLS=UseSolLS, device="cuda")
      CompileImagesToVideo(output_dir, videoname, fps=3)

    else: raise ValueError("Must choose 'B' or 'O'.")
