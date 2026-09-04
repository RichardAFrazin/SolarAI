#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:52:07 2026

@author: rfrazin

my highway.mp4 video comes from the pixbay.com website:
   pixabay.com/videos/car-road-transportation-vehicle-2165/


"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import torch
import VideoUtils as VU

mp4file = "highway1.mp4" # "highway2.mp4"
cen1 = (200,381); cen2 = (200, 262); npix=80
setup = {'sz': npix, 'n_ang': 26, 'n_rays': int(np.round(2*np.sqrt(2)*npix)) }



#This calculates the matrix corresponding to the projection at input angle
#  angle - the intput angle is in radian units; any real number is ok.
#    angle = 0 corresponds to a horizontal projection with the screen parallel to the y-axis,
#        and the rays parallel to the x-axis.
def ProjectionSubMatrix(angle, setup=setup):
    sz = setup['sz']  # Tamaño de la imagen (suponiendo que es cuadrada)
    M = setup['n_rays']  # Número de rayos por ángulo
    d_ray = np.sqrt(2) * sz / M  # Distancia entre rayos
    A_i = np.zeros((M, sz**2))
    def PiMod(s):
       t = s % (2*np.pi)
       if t > np.pi: t -= 2*np.pi
       return t
    angle = PiMod(angle)
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
#  AnglesTimes can also be a zip object
#It optionally returns the projection matrix corresponding to all of the angles at a fixed time, which permits static reconstruction.
#The time value in each tuple corresponds to a frame number.  See the VideoInerpolate function
#The angle (between -pi and pi) in each angle is in radian units
#video - 3D np.array in which the 0th index corresponds to the frames.  See VideoInterpolate
#ReturnProjMat - if True, return the static projection matrix for the angles in AnglesTimes
#ReturnBackProjs - if True, return a 3D array of backprojected images instead of the projections themselves
def ProjectionTimeSeries(AnglesTimes, video, setup=setup, ReturnProjMat=False, ReturnBackProjs=False):
    if video.ndim != 3:  raise ValueError("video must be a 3D array.")
    if ReturnProjMat and ReturnBackProjs:
       raise ValueError("ReturnProjMat and ReturnBackProjs cannot both be True simultaneously.")
    AnglesTimes = list(AnglesTimes)  #  this will do nothing if it is already a list, but it turns into a list if it is a zip object
    times = [] ;  projs = []; backprojs = []   # stuff will be collected here
    A = [];  # if desired, the proj matrix will be collected here
    for k, tup in enumerate(AnglesTimes):  # k is the interpolated video index
       times.append(tup[1])
    vid = VU.VideoInterpolate(times, video)
    nrowsA = 0
    for k, tup in enumerate(AnglesTimes):
        A_i = ProjectionSubMatrix(tup[0], setup=setup)
        proj = A_i@vid[k,:,:].reshape((A_i.shape[1],))
        projs.append( proj )
        if ReturnBackProjs:
           backprojs.append( (A_i.T@proj).reshape((vid[k,:,:].shape))  )
        elif ReturnProjMat:
          nrowsA += A_i.shape[0]
          A.append(A_i)
    if ReturnProjMat:
       A = np.array(A).reshape((nrowsA, A_i.shape[1]))
       return( (projs, A) )
    elif ReturnBackProjs:
       return( np.array(backprojs) )
    else:
       return A

#%%  Least-squares solution stuff

#This creates a representation of the $\nabla$ operator in sparse matrix format for regularization.
#The solution being regularized corresponds to an N-by-N image.
def Nabla_sparse(N):
    main_diag = - np.ones(N);
    main_diag[-1] = 0.  # there is no neighbor the right
    upper_diag = np.ones(N-1)

    D = sp.diags([main_diag, upper_diag], [0, 1], shape=(N, N), format='csr')
    Id = sp.eye(N, format='csr')
    #extend this to 2D
    Dx = sp.kron(Id, D, format='csr')
    Dy = sp.kron(D, Id, format='csr')
    Nabla = sp.vstack([Dx, Dy] ,format='csr')
    return Nabla

def ID_sparse(N):  # this image is assume to be NxN pixels
   return sp.eye(N*N, format='csr')

#This performs a regularized static reconstruction via least-squares solution
#   The projection data are acquired as the video evolves in time.
#video - np.array of shape (n_frames, 80. 80)
#ProjMats - list of projection matrices corresponding to the observed angles
#    can be in a scipy.sparse format, but not a Torch.tensor
#ProjTimes - list of times (video frame units), must be >= 0 and <= n_frames-1
#   Note:  len(ProjMat) must be equal to len(ProjTimes)
#RegFcn - either None, 'Nabla_sparse', or 'ID_sparse'
#regparam - positive multiplier of the regularization matrix
#ShowSolver - set to True to see the output of the least-squares solver
def StaticReconstruction(video, ProjMats, ProjTimes, RegFcn='Nabla_sparse', regparam=0.01, UseTorch=False, ShowSolver=False):
   if RegFcn is not None:
      if RegFcn not in ['ID_sparse','Nabla_sparse']:
         raise ValueError("Invalid RegFcn string.")
   if len(ProjMats) != len(ProjTimes):
      raise ValueError("The input lists ProjMats and ProjTimes must have the same length.")
   if video.ndim != 3:  raise ValueError("video must be a 3D array.")
   n_frames = video.shape[0]
   ProjTimes = np.array(ProjTimes)
   if not all(ProjTimes >=0.) and all(ProjTimes <= n_frames-1):
      raise ValueError("All ProjTimes values must be >= 0 and <= n_frames-1.")

   projs = []
   vid = VU.VideoInterpolate(ProjTimes, video)
   for k in range(len(ProjTimes)):
      projs.append( ProjMats[k]@vid[k,:,:].reshape((80*80,)) )
      if k == 0 :
         A = sp.csr_matrix(ProjMats[0])
      else:
         A = sp.vstack( (A, sp.csr_matrix(ProjMats[k])) )
   projs = np.array(projs).reshape( (len(projs[0])*len(projs),) )

   Reg = None  # regularization matrix
   if RegFcn == 'Nabla_sparse':
      Reg = regparam*Nabla_sparse(80)
   elif RegFcn == 'ID_sparse':
      Reg = regparam*ID_sparse(80)
   if Reg is not None:
      A = sp.vstack((A, Reg))
      projs = np.hstack( (projs, np.zeros(Reg.shape[0])) )

   if UseTorch: # apply ConjGrad in PyTorch to get the least-squares solution
      A = A.tocsr()  # Make sure it's in sp.CSR format
      A_torch = torch.sparse_csr_tensor(  # put it on the GPU
            crow_indices=torch.from_numpy(A.indptr).long(),
            col_indices=torch.from_numpy(A.indices).long(),
            values=torch.from_numpy(A.data).float(),
            size=A.shape, device="cuda")
      projs_torch = torch.from_numpy(projs).float().to("cuda")

      x = ConjGradLeastSq_Torch(A_torch, projs_torch, device=A_torch.device, max_iter=100, tol=1.e-6)
      return x.cpu().numpy().reshape((80, 80))

   else:
      LSMR = sp.linalg.lsmr # iterative least-squares solution via the LSMR method
      x, istop, itn, normr, normar, norma, conda, normx = LSMR(A, projs,
      atol=1.e-6, btol=1.e-6, maxiter=100, show=ShowSolver)  # atol - matrix resid, btol - y resid
      return x.reshape((80,80))

@torch.no_grad()  # no need for computation graph
def ConjGradLeastSq_Torch(A, b, device, max_iter=100, tol=1e-6):
    x = torch.zeros(A.shape[1], dtype=torch.float32, device=device)
    r = b - torch.mv(A, x)  # .mv is matrix-vector multiplication
    p = torch.mv(A.t(), r)
    d = p.clone()

    norm_g_old = torch.sum(p ** 2)

    for i in range(max_iter):
        Ad = torch.mv(A, d)
        alpha = norm_g_old / torch.sum(Ad ** 2)

        x = x + alpha * d
        r = r - alpha * Ad

        p = torch.mv(A.t(), r)  # matrix vector multiplication
        norm_g_new = torch.sum(p ** 2)

        if torch.sqrt(norm_g_new) < tol:
            break

        beta = norm_g_new / norm_g_old
        d = p + beta * d
        norm_g_old = norm_g_new

    return x.squeeze()


#%% demo routines


def demo_static(video, regparam=0.01):
   print("Least-Squares Reconstruction of a static object.")
   n_ang = 31
   angs = np.linspace(0, np.pi*(n_ang-1)/n_ang , n_ang)
   #times = np.linspace(0, 38.5, n_ang)
   times = 7*np.ones((n_ang,))   #frame number 20
   ProjMats = []

   for ang in angs:
      ProjMats.append( sp.csr_matrix(ProjectionSubMatrix(ang, setup=setup)) )
   x = StaticReconstruction(video, ProjMats, times, RegFcn='Nabla_sparse', regparam=regparam, UseTorch=True, ShowSolver=False);
   return x.reshape((80,80))

def demo_dynamic(video, regparam=0.1):
   print("Static Least-Squares Reconstruction of a time-dependent object.")
   n_ang = 31
   angs = np.linspace(0, np.pi*(n_ang-1)/n_ang , n_ang)
   times = np.linspace(2, 12, n_ang)
   ProjMats = []
   for ang in angs:
      ProjMats.append( sp.csr_matrix(ProjectionSubMatrix(ang, setup=setup)) )
   x = StaticReconstruction(video, ProjMats, times, RegFcn='Nabla_sparse', regparam=regparam, UseTorch=True, ShowSolver=False);
   return x.reshape((80,80))


#%% run demos
if __name__ == "__main__":
   lilvid = VU.vid1[1121:1121+40,:,:]
   x_static = demo_static(lilvid, regparam=0.02)
   x_dynamic = demo_dynamic(lilvid, regparam=0.05)
   plt.figure(); plt.imshow(lilvid[2,:,:],cmap='coolwarm');plt.colorbar();plt.title('frame 2, True');
   plt.figure(); plt.imshow(lilvid[7,:,:],cmap='coolwarm');plt.colorbar();plt.title('frame 7, True');
   plt.figure(); plt.imshow(lilvid[12,:,:],cmap='coolwarm');plt.colorbar();plt.title('frame 12, True');
   plt.figure(); plt.imshow(x_static,cmap='coolwarm');plt.colorbar();plt.title('frame 7, static recon');
   plt.figure(); plt.imshow(x_dynamic,cmap='coolwarm');plt.colorbar();plt.title('frames 2-12, dynamic recon');
