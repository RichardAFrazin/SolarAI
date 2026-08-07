#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 13:05:24 2026

@author: rfrazin
"""

import numpy as np
import torch
import matplotlib.plyplot as plt
import Tomo2D_UNet as UN
import ProjectionUtils2D as PU


"""
This creates a training set for the UNet.  The idea is for the UNet to learn to
  extrapolate in time.  The input video ("video") is divided into chunks of
  "nFramesOut" with a stride in integer units specifed by "stride", which can
  be None, in which case it is set to nFramesOut//2.
The UNet output is "nFramesOut" 80x80 images, which are compared to the corresponding
  video images in the loss function.  This function collects these "truth" images.
The UNet input is a set of "nAnglesObs" 80x80 Radon backprojections at angles evenly
  distributed over (0, pi).  The projections are obtained over the time interval
  specifed by "StartEndObs", which is a tuple (list, array) of length 2 containing
  the integer frame numbers between 0 and "nFramesOut"-1.

"""
def TrainingDataSet(nFramesOut, nAnglesObs, StartEndObs, stride=None, video=PU.vid1):
   if stride is None: stride = nFramesOut//2
   stride = int(stride)
   if video.ndim != 3:
      raise ValueError(f"video.ndim must be 3, but it is {video.ndim}.")
   StartEndObs = np.array(StartEndObs)
   if (len(StartEndObs) != 2) or (not StartEndObs[0].is_integer()) or (not StartEndObs[1].is_integer()):
      raise ValueError("StartEndObs must have length 2 with integer values.")
   if (video.shape[1] != 80) or (video.shape[2] != 80):
      raise ValueError("video must size (Nframes, 80, 80).")

   #The projection angles are the same for each sample, so only compute them once
   angles = np.linspace(0, np.pi*nAnglesObs/(nAnglesObs-1), nAnglesObs)
   projmats = []
   for angle in angles:
      projmats.append(PU.ProjectionSubmatrix(angle, setup=PU.setup))

   samples = []  # list of input/output tuples

   VideoEnd = False # True signals the end of the video
   k = -1  #sample number
   while not VideoEnd:
      k += 1
      truthvid = video[k*stride : k*stride + nFramesOut, :, :]  # truth video
      bpvid = []  # backprojection video
      times =  np.linspace(StartEndObs[0], StartEndObs[1], nAnglesObs)
      #  get the portion of truthvid corresponding to the observation times
      vidobs = PU.VideoInterpolate(times, truthvid)
      for i in range(vidobs.shape[0]):
         bpvid.append( projmats[i].T@projmats[i]@vidobs[i].reshape((80*80,))  )
      samples.append( (np.array(bpvid), truthvid)  )
      if (k+1)*stride + nFramesOut >= video.shape[2]:  #terminate while loop
         VideoEnd=True

   return samples
