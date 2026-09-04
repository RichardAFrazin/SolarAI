#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:14:55 2026
by Richard Frazin    traffic2_file = "highway2.mp4"


These are tools for manipulating video inputs to 2D
   tomography experiments.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator as Akima
import scipy.sparse as sp
import cv2

npix=80  #  this project is setup for 80x80 image sequences


def Loadmp4(filename):  # load video and convert it to greyscale
   cap = cv2.VideoCapture(filename)
   if not cap.isOpened():
      print("Can't open video file")
      exit()
   frames = []
   while cap.isOpened(): # make greyscale frames out of color frames
      ret, frame = cap.read()
      if not ret: break # end of video
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      frames.append(gray_frame)
   cap.release()
   return(np.array(frames))


#this uses SciPy's Akima1DInterpolator to evaluate a video at times between the frame numbers
#  video - the video to be evalued.  The first dimension is the frames. E.g., (3000, 256,256) for
#
#  times specifies the frame numbers, or fractions thereof, at which the video will be evaluated.
#     times can be a single float or integer, list or 1D array of times.
#
def VideoInterpolate(times, video):
   tt = np.atleast_1d(times)  # converts times to an np.array
   if video.ndim != 3:
      raise ValueError(f"Video shape is {video.shape}.  Video must be a 3D array.")
   if not ( all(tt >= 0.) and all(tt <= video.shape[0]-1) ):
      raise ValueError(f"Input time = {tt}.  All times must be at least zero and less than the (number of frames in the video)-1.")
   frames = np.arange(video.shape[0])
   interpolator = Akima(frames, video, axis=0)
   res = interpolator(tt)
   if np.isscalar(times):  # drop the unwanted dimension from the output
      return(res[0])
   return( res )

def IntegerRebin2DArray(array, new_shape): #rebin by averaging over an integer number pixels
   shape = (new_shape[0], array.shape[0]//new_shape[0], #4  dimensions
            new_shape[1], array.shape[1]//new_shape[1])
   return( array.reshape(shape).mean(3).mean(1) )

def CV2Rebin2DArray(array, new_shape):
   newar = cv2.resize(array, new_shape, interpolation=cv2.INTER_CUBIC)
   return newar

#%%

if __name__ == "__main__":

   traffic1_file = "highway1.mp4";
   t1_cen1 = (200,381); t1_cen2 = (200, 262)  # set extraction location


   t1_lgvid = Loadmp4(traffic1_file); # has shape (3000,720,1280)
   t1vid = []
   for k in range(t1_lgvid.shape[0]):
      t1vid.append( IntegerRebin2DArray(t1_lgvid[k,:,:],(360,640))  )
   t1vid = np.array(t1vid).astype('float')
   for k in range(t1vid.shape[0]):
      t1vid[k,:,:] = t1vid[k,:,:]/t1vid[k,:,:].max()
   t1_vid1 = t1vid[:,t1_cen1[0]-npix//2:t1_cen1[0]+npix//2, t1_cen1[1]-npix//2:t1_cen1[1]+npix//2]
   t1_vid2 = t1vid[:,t1_cen2[0]-npix//2:t1_cen2[0]+npix//2, t1_cen2[1]-npix//2:t1_cen2[1]+npix//2]
   del(t1_lgvid)

#%%

   traffic2_file = "highway2.mp4"
   t2newshape = (170,300)  # this scaling comes from comparing the distance between the road stripes in the traffic1 and traffic2 video at the acquisition row
   t2_cen1 = (148,169); t2_cen2 = (148, 259)
   t2_lgvid = Loadmp4(traffic2_file); # has shape (3000,720,1280)
   t2vid = []
   for k in range(t2_lgvid.shape[0]):
      t2vid.append( CV2Rebin2DArray(t2_lgvid[k,:,:],(t2newshape[1],t2newshape[0]) ) )
   t2vid = np.array(t2vid).astype('float')
   for k in range(t2vid.shape[0]):
      t2vid[k,:,:] = t2vid[k,:,:]/t2vid[k,:,:].max()
   t2_vid1 = t2vid[:,t2_cen1[0]-npix//2:t2_cen1[0]+npix//2, t2_cen1[1]-npix//2:t2_cen1[1]+npix//2]
   t2_vid2 = t2vid[:,t2_cen2[0]-npix//2:t2_cen2[0]+npix//2, t2_cen2[1]-npix//2:t2_cen2[1]+npix//2]
   del(t2_lgvid)
