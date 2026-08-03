#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:36:17 2026

@author: rfrazin
"""

import numpy as np

# This creates a random image from a 2D Gaussian process (GP)
# sz - image is sz-by-sz pixel
# scale - scale length (in pixels) for isotropic covariance fcn
# SqrtCov - This is the Cholesky square-root of the sz**2-by-sz**2 covariance matrix
#     - supplying SqrtCov removes all of the significant computation from this function.
#     - when it is supplied, the 'scale' input value is ignored.
# return_SqrtCov , returns the SqrtCov matrix when True
# returns the random image

def ImageFromGP(sz=80, scale=20.0, SqrtCov=None,return_SqrtCov=False):
    if return_SqrtCov:
       if SqrtCov is not None:
          raise ValueError("If you want to return the square-root of the covariance matrix, set SqrtCov to None.")
    if SqrtCov is None:  # Calculate covariance matrix and its sqrt
       x = np.arange(sz)
       y = np.arange(sz)
       X, Y = np.meshgrid(x, y)
       # Aplatir la grille pour obtenir une liste de coordonnées (N, 2) où N = sz * sz
       coords = np.column_stack((X.ravel(), Y.ravel()))
       # 2. Calcul des distances euclidiennes absolues entre tous les couples de pixels
       # Utilisation du broadcasting pour une matrice de distance de taille (N, N)
       dists = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=-1))
       # 3. Application de la fonction de covariance isotrope exponentielle
       # K(d) = exp(-d / lambda)
       K = np.exp(-dists / scale)
       # Ajout d'une petite perturbation (nugget) sur la diagonale pour la stabilité numérique
       K += 1e-8 * np.eye(sz**2)
       # 4. Décomposition de Cholesky (K = L @ L.T) to get matrix square-root
       L = np.linalg.cholesky(K)
    else:  # SqrtCov is supplied
       L = SqrtCov
       if L.shape != (sz**2, sz**2):
          raise ValueError(f"The matrix supplied for SqrtCov should have shape {(sz**2,sz**2)}, but it is {L.shape} .")
    z = np.random.normal(size=sz**2)
    img = (L @ z).reshape((sz, sz))  # resulting Gaussian output image
    offset = np.min(img[img < 0.])  # make output non-negative
    img -= offset
    if return_SqrtCov:
       return( (img, L) )
    else:
       return img
