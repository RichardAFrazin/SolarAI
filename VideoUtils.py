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
import matplotlib.animation as anim
import os
from scipy.interpolate import Akima1DInterpolator as Akima
import cv2

npix=80  #  this project is setup for 80x80 image sequences


#%%extract the portion of a .mp4 video between times t1 and t2 (units: seconds)
def ExtractVideoPortion(VidFileIn, VidFileOut, t1, t2, target_size=None, grayscale=True):
   cap = cv2.VideoCapture(VidFileIn)
   if not cap.isOpened():
         raise ValueError("Invalid Input Video File.")

   def CountFrames():  # count the images in the video.  Last resort in case of a problem reading the header information.
        print("Counting frames in the video. A 10 min video w/ 60 fps takes about 30 s.")
        count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  #start at first image
        while True:
            ret, _ = cap.read()  # On lit l'image sans la stocker en mémoire (_)
            if not ret:
                break
            count += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # rewind the video
        return count

   fps = cap.get(cv2.CAP_PROP_FPS) # may not work due to a header problem
   total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
   # total_frames = CountFrames()
   total_time = total_frames/fps
   if not (0 <= t1 < t2 <= total_time):
      raise ValueError(f"Invalid cut times.  Video duration is {total_time} seconds.")

   orig_width =  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   out_width = target_size[0]; out_height = target_size[1]

   frame_start = int(t1*fps);
   frame_end = int(t2*fps)

   fourcc = cv2.VideoWriter_fourcc(*'mp4v')

   out = cv2.VideoWriter(VidFileOut, fourcc, fps, (out_width, out_height), isColor=not grayscale)
   cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
   current_frame = frame_start
   while current_frame <= frame_end:
      ret, frame = cap.read()
      if not ret:
         break
      if grayscale:
         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      if target_size is not None:
         frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
      else: pass

      out.write(frame)
      current_frame += 1

   cap.release(); out.release()
   cv2.destroyAllWindows()
   print("Successful Extraction.")
#%%

#The calling function should create a variable for the animation object ("ani")
# video - a 3D array with first dimension being time (frame numbers)
# frame_interval - time per frame in milliseconds
def AnimateNpArray(video, frame_interval=15, repeat=False, cmap='coolwarm'):
   if video.ndim != 3:
      raise ValueError(f"Video shape is {video.shape}.  Video must be a 3D array.")

   fig, ax = plt.subplots();
   ax.axis('off'); # makes a cleaner presentation
   #show first image
   im = ax.imshow(video[0], cmap=cmap, vmin=video.min(), vmax=video.max())

   def update(frame_index):
      im.set_data(video[frame_index])
      return [im]


   ani = anim.FuncAnimation(fig, update, frames=video.shape[0], blit=True,
                            interval=frame_interval,repeat=repeat);

   plt.show() #launch video
   return ani

#%%
def Loadmp4(filename):  # load video and convert it to greyscale
   print("This is too memory intensive for large videos.")
   cap = cv2.VideoCapture(filename)
   if not cap.isOpened():
      raise ValueError("Can't open video file")
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
   if video.shape[0] == 1:  #only 1 frame in the video
      res = np.zeros((times.shape[0],video.shape[1],video.shape[2]))
      for kt in range(times.shape[0]):
         res[kt,:,:] = video[0,:,:]
      return( res )
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
   #to see the keys of this dict-like object returned by this fcn,
   #   use print(list(vids.keys()))
def Load80x80Videos(subject='Ducks'):
   subjects = ['Ducks','Traffic']
   if subject not in subjects:
      raise ValueError(f"kwarg 'subject' must be one of: {subjects}")
   duckvid80x80file    = "DuckVideo80x80Arrays.npz"
   if not os.path.isfile(duckvid80x80file):
      raise FileNotFoundError(f"Cannot find {duckvid80x80file}")
   trafficvid80x80file = "TrafficVideo80x80Arrays.npz"
   if not os.path.isfile(trafficvid80x80file):
      raise FileNotFoundError(f"Cannot find {trafficvid80x80file}")
   if subject == 'Ducks':
      vids = np.load(duckvid80x80file)
   elif subject == 'Traffic':
      vids = np.load(trafficvid80x80file)
   else:
      vids = None
   return vids



#%%

   if False:

      traffic1_file = "highway1.mp4";
      t1_cen1 = (200,381); t1_cen2 = (200, 262)  # set extraction location


      t1_lgvid = Loadmp4(traffic1_file); # has shape (3000,720,1280)
      t1vid = []
      for k in range(t1_lgvid.shape[0]):
         #t1vid.append( IntegerRebin2DArray(t1_lgvid[k,:,:],(360,640))  )
         t1vid.append( CV2Rebin2DArray(t1_lgvid[k,:,:] , (640,360)) )
      t1vid = np.array(t1vid).astype('float')
      for k in range(t1vid.shape[0]):
         t1vid[k,:,:] = t1vid[k,:,:]/t1vid[k,:,:].max()
      t1_vid1 = t1vid[:,t1_cen1[0]-npix//2:t1_cen1[0]+npix//2, t1_cen1[1]-npix//2:t1_cen1[1]+npix//2]
      t1_vid2 = t1vid[:,t1_cen2[0]-npix//2:t1_cen2[0]+npix//2, t1_cen2[1]-npix//2:t1_cen2[1]+npix//2]
      del(t1_lgvid)

      traffic2_file = "highway2.mp4"
      t2newshape = (170,300)  # this scaling comes from comparing the distance between the road stripes in the traffic1 and traffic2 video at the acquisition row
      t2_cen1 = (130,179); t2_cen2 = (130, 259)
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

      FourStack = lambda k : np.hstack( (np.vstack((t1_vid1[k,:,:],t2_vid1[k,:,:])) , np.vstack((t1_vid2[k,:,:],t2_vid2[k,:,:])) ) )

#%%

   if False:  # get 4 videos, each with 80x80 pixels
      duckvidfile = '../../DuckVideos/MVI_0011_cut2_.MP4'
      dv = Loadmp4(duckvidfile)

      #80x80 pixel selections
      r1 = [60, 140,  20, 100]; dv1 = dv[:, r1[0]:r1[1], r1[2]:r1[3]].astype('float')
      r2 = [70, 150, 100, 180]; dv2 = dv[:, r2[0]:r2[1], r2[2]:r2[3]].astype('float')
      r3 = [40, 120, 180, 260]; dv3 = dv[:, r3[0]:r3[1], r3[2]:r3[3]].astype('float')
      r4 = [90, 170, 240, 320]; dv4 = dv[:, r4[0]:r4[1], r4[2]:r4[3]].astype('float')
      for kt in range(dv.shape[0]):
         norm = np.max(dv[kt,:,:])
         dv1[kt,:,:] /= norm; dv2[kt,:,:] /= norm; dv3[kt,:,:] /= norm; dv4[kt,:,:] /= norm

   if False:
#%%    #run some static recons to get a feel for the temporal varation
       # an acquisition time of 16 frames produces images that are sifnificantly
       #    compromised by the time variation, but not complete garbage.
      import ProjectionUtils2D as PU
      vid = dv3
      n_ang = 40
      central_frames = [ 177, 8059, 17998, 22664, 13500,  2344]
      acq_time = [0,2,4,10,12, 16,20,30]  #length of time (frame units) to acquire projections (even ints)
      angles = np.linspace(0, np.pi*(n_ang-1)/n_ang, n_ang)
      ProjMats = []
      for ang in angles:
         ProjMats.append(PU.ProjectionSubMatrix(ang))

      images_tru = [] # true images
      recon_errs  = np.zeros( (len(central_frames), len(acq_time)) )
      for nc in range(len(central_frames)):
         images_tru.append(vid[central_frames[nc],:,:])
      for nc in range(len(central_frames)):
         for nt in range(len(acq_time)):
            times = np.linspace(0, acq_time[nt], n_ang)
            fstart = int(central_frames[nc] - acq_time[nt]/2) #first frame
            fend   = int(fstart + acq_time[nt])  # last frame
            if fstart < 0 or fend > vid.shape[0]:
               raise ValueError("Invalid Video Frames Requested.")
            vv = vid[fstart : fend + 1,:,:]
            rec = PU.StaticReconstruction(vv, ProjMats, times, RegFcn='Nabla_sparse', regparam=0.05, UseTorch=False, ShowSolver=False)
            rms = np.sqrt(np.mean( (images_tru[nc] - rec)**2 ))
            recon_errs[nc, nt] = rms

      plt.figure(); plt.plot(acq_time,recon_errs.T)
