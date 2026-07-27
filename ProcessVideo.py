#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:52:07 2026

@author: rfrazin

my highway.mp4 video comes from the pixbay.com website:
   pixabay.com/videos/car-road-transportation-vehicle-2165/


"""

import numpy as np
from scipy.interpolate import Akima1DInterpolator as Akima
import cv2

mp4file = "highway.mp4"; cen1 = (200,381); cen2 = (200, 262); npix=80
setup = {'sz': npix, 'n_ang': 26, 'n_rays': int(np.round(2*np.sqrt(2)*npix)) }
#%%

#This calculates the matrix corresponding to the projection at input angle
#  The intput angle is in radian units.
#  angle = 0 corresponds to a horizontal projection with the screen parallel to the y-axis,
#        and the rays parallel to the x-axis.
def ProjectionSubMatrix(angle, setup=setup):
    if (angle < - np.pi) or (angle > np.pi):
       raise ValueError("angle must between -pi and pi.")
    sz = setup['sz']  # Tamaño de la imagen (suponiendo que es cuadrada)
    M = setup['n_rays']  # Número de rayos por ángulo
    d_ray = np.sqrt(2) * sz / M  # Distancia entre rayos
    A_i = np.zeros((M, sz**2))

    #  These vertical and horizontal lines define the pixel boundaries.
    vertical_lines = np.linspace(-sz//2, sz//2, sz+1)
    horizontal_lines = vertical_lines
    t_max = np.sqrt(2) * sz / 2  # El valor máximo de |t| (la distancia más larga desde el centro de la imagen)
    eps_ang = 1.0e-5  # Tolerancia para tratar los ángulos horizontales/verticales
    VerticalRays = False; HorizontalRays = False
    if abs(angle - np.pi) < eps_ang or abs(angle) < eps_ang or (angle + np.pi) < eps_ang :  # horizontal projection
      HorizontalRays = True
    if abs(angle - np.pi/2) < eps_ang or abs(angle + np.pi/2) < eps_ang: # vertical projection
      VerticalRays = True
    sin = np.sin(angle); cos = np.cos(angle)
    s = np.array([cos, sin]) # unit vector along rays
    for m in range(M):  # m is the ray index
       p0_m = (m - M // 2)*d_ray*np.array([- sin, cos])  # central point of ray
       intersections = []  # list of intersection "times" with the vertical and horizontal grid lines
       for x in vertical_lines:
          if VerticalRays:
             continue
          t = (x - p0_m[0]) / s[0]
          if -t_max < t < t_max:
               intersections.append(('v', x, t))
       for y in horizontal_lines:
            if HorizontalRays:
               continue
            t = (y - p0_m[1]) / s[1]
            if -t_max < t <=t_max:
               intersections.append(('h', y, t))
       intersections.sort(key = lambda x: x[2])  # sort intersections
       for j in range(1, len(intersections)):
          prev_t = intersections[j-1][2]  # Valor t de la intersección anterior
          curr_t = intersections[j][2]  # Valor t de la intersección actual
          t_mid = 0.5 * (prev_t + curr_t)  # Valor t medio entre las intersecciones
          q = p0_m + s*t_mid
          pixel_x = int(np.round(q[0] + sz // 2))  # Convertir coordenada x a índice de píxel
          pixel_y = int(np.round(q[1] + sz // 2))  # Convertir coordenada y a índice de píxel
          if 0 <= pixel_x < sz and 0 <= pixel_y < sz:  # Asegurarse de que los índices estén dentro del rango
              A_i[m, pixel_y * sz + pixel_x] = curr_t-prev_t  # Asignar valor en la matriz A
    A_i /= sz  # normalization by the typical ray length
    return(A_i)

#%%

#this uses SciPy's Akima1DInterpolator to evaluate a video at times between the frame numbers
#  video - the video to be evalued.  The first dimension is the frames. E.g., (3000, 256,256) for
#
#  times specifies the frame numbers, or fractions thereof, at which the video will be evaluated.
#     times can be a single float or integer, list or 1D array of times.
#
def VideoInterpolate(times, video):
   tt = np.atleast_1d(times)  # converts times to an np.array
   if video.ndim != 3:
      raise ValueError("video must be a 3D array.")
   if not ( all(tt >= 0.) and all(tt <= video.shape[0]-1) ):
      raise ValueError("all times must be at least zero and less than the (number of frames in the video)-1.")
   frames = np.arange(video.shape[0])
   interpolator = Akima(frames, video, axis=0)
   res = interpolator(tt)
   if np.isscalar(times):  # drop the unwanted dimension from the output
      return(res[0])
   return( res )

#This accumulates the submatrices corresponding to the projection angles
def ProjectionMatrix(setup=setup):
    n_ang = setup['n_ang']  # setup  projection angles
    angles = np.linspace(np.pi / (2 * n_ang), np.pi * (1 - 1 / (2 * n_ang)), n_ang)
    A = [] # the submatrices will be appended to A
    nrowsA = 0 # number of rows A will eventually have
    for i, angle in enumerate(angles):  # get the submatrix for each projection angle
        A_i = ProjectionSubMatrix(angle, setup=setup)
        A.append(A_i)
        nrowsA += A_i.shape[0]
    A = np.array(A).reshape((nrowsA, A_i.shape[1]))
    return(A)

#This returns the set of projections corresponds to a given set of tuples, specified in the
#  in AnglesTimes input variable, which is a list of tuples in the form (time, angle).
#It optionally returns the projection matrix corresponding to all of the angles at a fixed time, which permits static reconstruction.
#The time value in each tuple corresponds to a frame number.  See the VideoInerpolate function
#The angle (between -pi and pi) in each angle is in radian units
#video - 3D np.array in which the 0th index corresponds to the frames.  See VideoInterpolate
#ReturnProjMat - set to True in order to get the static projection matrix for the angles in AnglesTimes
def ProjectionTimeSeries(AnglesTimes, video, setup=setup, ReturnProjMat=False):
    if video.ndim != 3:
      raise ValueError("video must be a 3D array.")
    times = [] ;  projs = []  # projections will be collected here
    A = [];  # if desired, the proj matrix will be collected here
    for k, tup in enumerate(AnglesTimes):  # k is the interpolated video index
       times.append(tup[1])
    vid = VideoInterpolate(times, video)
    nrowsA = 0
    for k, tup in enumerate(AnglesTimes):
      A_i = ProjectionSubMatrix(tup[0], setup=setup)
      projs.append( A_i@vid[k,:,:].reshape((A_i.shape[1],)) )
      if ReturnProjMat:
          nrowsA += A_i.shape[0]
          A.append(A_i)
    if ReturnProjMat:
       A = np.array(A).reshape((nrowsA, setup['sz']**2))
       return( (projs, A) )
    else:
       return A

def rebin2Darray(array, new_shape): #rebin by averaging over 2x2 pixels
   shape = (new_shape[0], array.shape[0]//new_shape[0], #4  dimensions
            new_shape[1], array.shape[1]//new_shape[1])
   return( array.reshape(shape).mean(3).mean(1) )

def Loadmp4(filename=mp4file):
   cap = cv2.VideoCapture(mp4file)
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
#%%

lgvid = Loadmp4(); #lgvid has shape (3000,720,1280)
vid = []
for k in range(lgvid.shape[0]):
   vid.append( rebin2Darray(lgvid[k,:,:],(360,640))  )
vid = np.array(vid)
del(lgvid)
#normpix = (199,430) # a pixel on the white line
for k in range(vid.shape[0]):
   vid[k,:,:] /= vid[k,:,:].max()
vid1 = vid[:,cen1[0]-npix//2:cen1[0]+npix//2, cen1[1]-npix//2:cen1[1]+npix//2]
vid2 = vid[:,cen2[0]-npix//2:cen2[0]+npix//2, cen2[1]-npix//2:cen2[1]+npix//2]

def demo():
   lilvid = vid1[1121:1121+40,:,:]
   n_ang = 13
   angs = np.linspace(0, np.pi*(n_ang-1)/n_ang , n_ang)
   times = np.linspace(0, 39.5, n_ang)
   AngTimes = zip(angs, times)
   projs, Mat = ProjectionTimeSeries(AngTimes, lilvid, setup=setup, ReturnProjMat=False)
   return (projs, Mat)


#%%    ######## Junk Yard  ################


def _ProjectionMatrix(setup=setup):
    sz = setup['sz']  # Tamaño de la imagen (suponiendo que es cuadrada)
    n_ang = setup['n_ang']  # Número de ángulos de vista
    M = setup['n_rays']  # Número de rayos por ángulo
    d_ray = np.sqrt(2) * sz / M  # Distancia entre rayos

    # projection angles
    angles = np.linspace(np.pi / (2 * n_ang), np.pi * (1 - 1 / (2 * n_ang)), n_ang)

    # Crear la matriz A (n_ang * M x sz^2), inicializada en ceros
    A = np.zeros((n_ang * M, sz**2))

    # Definir las líneas de la cuadrícula (líneas horizontales y verticales)
    vertical_lines = np.linspace(-sz//2, sz//2, sz+1)  # Líneas verticales que definen los límites de los píxeles
    horizontal_lines = np.linspace(-sz//2, sz//2, sz+1)  # Líneas horizontales que definen los límites de los píxeles
    eps_ang = 1.0e-5  # Tolerancia para tratar los ángulos horizontales/verticales

    # Definir los límites de t
    t_max = np.sqrt(2) * sz / 2  # El valor máximo de |t| (la distancia más larga desde el centro de la imagen)

    for i, angle in enumerate(angles):  # rays are PARALLEL to the projection angle
        VerticalRays = False; HorizontalRays = False
        if abs(angle - np.pi) < eps_ang or abs(angle) < eps_ang :  # horizontal projection
           HorizontalRays = True
        if abs(angle - np.pi/2) < eps_ang or abs(angle + np.pi/2) < eps_ang: # vertical projection
           VerticalRays = True
        sin = np.sin(angle); cos = np.cos(angle)
        s = np.array([cos, sin]) # unit vector along rays

        for m in range(M):
            p0_m = (m - M // 2)*d_ray*np.array([- sin, cos])  # central point of ray
            intersections = []  # list of intersection "times" with the vertical and horizontal grid lines
            for x in vertical_lines:
               if VerticalRays:
                  continue
               t = (x - p0_m[0]) / s[0]
               if -t_max < t < t_max:
                    intersections.append(('v', x, t))
            for y in horizontal_lines:
                 if HorizontalRays:
                    continue
                 t = (y - p0_m[1]) / s[1]
                 if -t_max < t <=t_max:
                    intersections.append(('h', y, t))
            intersections.sort(key = lambda x: x[2])  # sort intersections

            # Ahora, para cada par de intersecciones consecutivas, actualizamos la matriz A
            for j in range(1, len(intersections)):
                prev_t = intersections[j-1][2]  # Valor t de la intersección anterior
                curr_t = intersections[j][2]  # Valor t de la intersección actual
                t_mid = 0.5 * (prev_t + curr_t)  # Valor t medio entre las intersecciones
                q = p0_m + s*t_mid
                pixel_x = int(np.round(q[0] + sz // 2))  # Convertir coordenada x a índice de píxel
                pixel_y = int(np.round(q[1] + sz // 2))  # Convertir coordenada y a índice de píxel
                if 0 <= pixel_x < sz and 0 <= pixel_y < sz:  # Asegurarse de que los índices estén dentro del rango
                    A[i*M + m, pixel_y * sz + pixel_x] = curr_t-prev_t  # Asignar valor en la matriz A
            A /= sz  # normalization by the typical ray length
    return A
