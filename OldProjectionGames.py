#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:41:12 2025
@author: rfrazin


This generates simple 2D images for deep learning experiments

"""

import numpy as np
import matplotlib.pyplot as plt
import torch



setupdict = {'sz':64, 'n_hubs':7, 'n_ang':7,'dist_metric':'l5',
             'proj_resolution':2}


def CreateDataSet(n_samp=1000, setup=setupdict, UseTorch=False):
    sz = setup['sz']
    A = ProjectionMatrix(setup=setup)
    Aplus = np.linalg.pinv(A)

    if not UseTorch:
        # NumPy branch
        targets = np.zeros((n_samp, sz, sz), dtype=np.float32)
        inputs = np.zeros((n_samp, sz, sz), dtype=np.float32)

        for k in range(n_samp):
            truth = CreateImage(setup=setup, UseTorch=False)  # true image
            proj = A @ truth.reshape((sz * sz,))
            meas = (Aplus @ proj).reshape((sz, sz))
            targets[k] = truth
            inputs[k] = meas
    else:  # PyTorch branch
        A_torch = torch.from_numpy(A).float()  # Convert A to a PyTorch tensor
        Aplus_torch = torch.from_numpy(Aplus).float()  # Convert Aplus to a PyTorch tensor

        targets = torch.zeros((n_samp, sz, sz), dtype=torch.float32)
        inputs = torch.zeros((n_samp, sz, sz), dtype=torch.float32)

        for k in range(n_samp):
            truth = CreateImage(setup=setup, UseTorch=True)  # true image
            truth_torch = torch.from_numpy(truth).float()  # Convert truth to a PyTorch tensor

            # Perform matrix multiplication using PyTorch
            proj = torch.matmul(A_torch, truth_torch.reshape((sz * sz,)))
            meas = torch.matmul(Aplus_torch, proj).reshape((sz, sz))

            # Store results in pre-allocated tensors
            targets[k] = truth_torch
            inputs[k] = meas

    return (inputs,targets)

def Distance(p1, p2, metric='l2'):
    """
    Calcula la distancia entre dos puntos p1 y p2, según la métrica especificada.
    Args:
    - p1: Tupla (p1x, p1y) del primer punto.
    - p2: Tupla (p2x, p2y) del segundo punto.
    - metric: Métrica para calcular la distancia. 'l1' para Manhattan,
              'l2' para Euclidiana, 'l0' para L0 (número de componentes no nulos),
              'l5' para L5 (p-norm para p=5).
    Returns:
    - La distancia entre los puntos p1 y p2 según la métrica indicada.
    """
    p1x, p1y = p1
    p2x, p2y = p2

    diff_x = abs(p2x - p1x)
    diff_y = abs(p2y - p1y)

    if metric == 'l2':
        # Distancia Euclidiana (L2)
        return np.sqrt(diff_x**2 + diff_y**2)
    elif metric == 'l1':
        # Distancia de Manhattan (L1)
        return diff_x + diff_y
    elif metric == 'l0':
        # Distancia L0: número de componentes no nulos
        return int(diff_x > 0) + int(diff_y > 0)
    elif metric == 'l5':
        # Distancia L5: p-norm para p=5
        return (diff_x**5 + diff_y**5) ** (1/5)
    else:
        raise ValueError(f"Métrica '{metric}' no soportada. Usa 'l1', 'l2', 'l0' o 'l5'.")
def DistanceTorch(p1, p2, metric='l2'):
    """
    Calculate the distance between two points p1 and p2 using PyTorch.

    Args:
    - p1: Tensor of shape (..., 2) representing the first point(s).
    - p2: Tensor of shape (..., 2) representing the second point(s).
    - metric: Metric for calculating the distance. 'l1' for Manhattan,
              'l2' for Euclidean, 'l0' for L0 (number of non-zero components),
              'l5' for L5 (p-norm for p=5).

    Returns:
    - Tensor of distances between p1 and p2.
    """
    diff = torch.abs(p1 - p2)  # Absolute difference between p1 and p2
    if metric == 'l2':
        # Euclidean distance (L2)
        return torch.norm(diff, p=2, dim=-1)
    elif metric == 'l1':
        # Manhattan distance (L1)
        return torch.sum(diff, dim=-1)
    elif metric == 'l0':
        # L0 distance: number of non-zero components
        return (diff != 0).sum(dim=-1)
    elif metric == 'l5':
        # L5 distance: p-norm for p=5
        return (diff ** 5).sum(dim=-1) ** (1/5)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def CreateImage(setup=setupdict, UseTorch=False):
    sz = setup['sz']  # Image size (sz x sz)
    n_hubs = setup['n_hubs']  # Number of hubs

    if not UseTorch:
        # NumPy implementation
        g = np.zeros((sz, sz))  # The image
        hub_center = []
        for k in range(n_hubs):
            hub_center.append(np.array([np.random.randint(0, sz), np.random.randint(0, sz)]))

        for kx in range(sz):
            for ky in range(sz):
                dist = np.zeros((n_hubs,))
                for kd in range(n_hubs):
                    dist[kd] = Distance((kx, ky), (hub_center[kd][0], hub_center[kd][1]), metric=setup['dist_metric'])
                hub = np.argmin(dist)
                g[ky, kx] = hub
        return g
    else:
        # PyTorch implementation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

        # Generate random hub centers
        hub_center = torch.randint(0, sz, (n_hubs, 2), device=device)  # Shape: (n_hubs, 2)

        # Create a grid of pixel coordinates
        kx, ky = torch.meshgrid(torch.arange(sz, device=device), torch.arange(sz, device=device), indexing='ij')
        pixel_coords = torch.stack([kx, ky], dim=-1).reshape(-1, 2)  # Shape: (sz*sz, 2)

        # Compute distances between each pixel and each hub
        diff = pixel_coords.unsqueeze(1) - hub_center.unsqueeze(0)  # Shape: (sz*sz, n_hubs, 2)
        dist = DistanceTorch(diff, torch.zeros_like(diff), metric=setup['dist_metric'])  # Use PyTorch Distance function

        # Assign each pixel to the nearest hub
        hub = torch.argmin(dist, dim=1)  # Shape: (sz*sz,)
        g = hub.reshape(sz, sz)  # Reshape to (sz, sz)

        return g.cpu().numpy()  # Return as NumPy array for compatibility



def ProjectionMatrix(setup=setupdict):
    sz = setup['sz']  # Tamaño de la imagen (suponiendo que es cuadrada)
    n_ang = setup['n_ang']  # Número de ángulos de vista

    # Número de rayos por ángulo (M), calculado con la fórmula ajustada
    M = int(np.ceil(setup['proj_resolution']*np.sqrt(2)*sz))  # Número de rayos por ángulo
    d_ray = np.sqrt(2) * sz / M  # Distancia entre rayos

    # Ángulos de vista (en radianes)
    angles = np.linspace(np.pi / (2 * n_ang), np.pi * (1 - 1 / (2 * n_ang)), n_ang)

    # Crear la matriz A (n_ang * M x sz^2), inicializada en ceros
    A = np.zeros((n_ang * M, sz**2))

    # Definir las líneas de la cuadrícula (líneas horizontales y verticales)
    vertical_lines = np.linspace(-sz//2, sz//2, sz+1)  # Líneas verticales que definen los límites de los píxeles
    horizontal_lines = np.linspace(-sz//2, sz//2, sz+1)  # Líneas horizontales que definen los límites de los píxeles

    # Definir los límites de t
    t_max = np.sqrt(2) * sz / 2  # El valor máximo de |t| (la distancia más larga desde el centro de la imagen)

    # Umbral de tolerancia para los ángulos cercanos a los valores singulares (0, pi/2, pi)
    epsilon = 1e-6  # Tolerancia para evitar los ángulos horizontales/verticales

    # Para cada ángulo de proyección
    for i, angle in enumerate(angles):
        # Verificar si el ángulo es cercano a los valores singulares
        if abs(abs(angle) - np.pi) < epsilon or abs(abs(angle) - np.pi/2) < epsilon:
            continue  # Si es casi horizontal o casi vertical, descartamos este ángulo

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Definir el origen del rayo, proyectado según el ángulo
        s = np.array([-sin_angle, cos_angle])  # Dirección del rayo (unit vector)

        # Para cada rayo asociado con este ángulo (cada uno de los M rayos)
        for m in range(M):
            p0_m = (m - M // 2) * d_ray * np.array([cos_angle, sin_angle])  # Desplazamiento del rayo

            # Calcular las intersecciones del rayo con las líneas horizontales y verticales
            intersections = []

            # Intersecciones con las líneas verticales (para cada x)
            if abs(angle - np.pi/2) > epsilon and abs(angle + np.pi/2) > epsilon:
              for x in vertical_lines:
                  t = (x - p0_m[0]) / s[0]
                  if -t_max < t < t_max:
                    intersections.append(('v', x, t))

            # Intersecciones con las líneas horizontales (para cada y)
            if abs(angle) > epsilon and abs(angle - np.pi) > epsilon:
              for y in horizontal_lines:
                  t = (y - p0_m[1]) / s[1]
                  if -t_max < t <=t_max:
                    intersections.append(('h', y, t))

            # Ordenar las intersecciones por el valor de t (distancia)
            intersections.sort(key = lambda x: x[2])

            # Ahora, para cada par de intersecciones consecutivas, actualizamos la matriz A
            for j in range(1, len(intersections)):
                prev_t = intersections[j-1][2]  # Valor t de la intersección anterior
                curr_t = intersections[j][2]  # Valor t de la intersección actual
                t_mid = 0.5 * (prev_t + curr_t)  # Valor t medio entre las intersecciones

                # Evaluar el punto q en el rayo en el parámetro t_mid
                q = p0_m + s*t_mid

                # Calcular los índices de píxel correspondientes a las coordenadas de q
                pixel_x = int(np.round(q[0] + sz // 2))  # Convertir coordenada x a índice de píxel
                pixel_y = int(np.round(q[1] + sz // 2))  # Convertir coordenada y a índice de píxel

                # Asegurarse de que los índices estén dentro del rango
                if 0 <= pixel_x < sz and 0 <= pixel_y < sz:
                    A[i * M + m, pixel_y * sz + pixel_x] = curr_t-prev_t  # Asignar valor en la matriz A
    return A


def __Distance(p1, p2, metric='l2'):
    """
    Calcula la distancia entre dos puntos p1 y p2, según la métrica especificada.

    Args:
    - p1: Tupla (p1x, p1y) del primer punto.
    - p2: Tupla (p2x, p2y) del segundo punto.
    - metric: Métrica para calcular la distancia. 'l1' para Manhattan,
              'l2' para Euclidiana, 'l0' para L0 (número de componentes no nulos),
              'l5' para L5 (p-norm para p=5).

    Returns:
    - La distancia entre los puntos p1 y p2 según la métrica indicada.
    """
    p1x, p1y = p1
    p2x, p2y = p2

    diff_x = abs(p2x - p1x)
    diff_y = abs(p2y - p1y)

    if metric == 'l2':
        # Distancia Euclidiana (L2)
        return np.sqrt(diff_x**2 + diff_y**2)
    elif metric == 'l1':
        # Distancia de Manhattan (L1)
        return diff_x + diff_y
    elif metric == 'l0':
        # Distancia L0: número de componentes no nulos
        return int(diff_x > 0) + int(diff_y > 0)
    elif metric == 'l5':
        # Distancia L5: p-norm para p=5
        return (diff_x**5 + diff_y**5) ** (1/5)
    else:
        raise ValueError(f"Métrica '{metric}' no soportada. Usa 'l1', 'l2', 'l0' o 'l5'.")

#This creates random piecewise constant images
def __CreateImage(setup = setupdict):
   sz = setup['sz']  # images will be sz-by-sz pixels

   n_hubs = setup['n_hubs'] # number of hubs
   g = np.zeros((sz,sz))  # the image
   hub_center = []
   for k in range(n_hubs):
      hub_center.append(np.array([np.random.randint(0,sz), np.random.randint(0,sz)]))

   dist = np.zeros((sz,sz,n_hubs))
   for kx in range(sz):
      for ky in range(sz):
         dist = np.zeros((n_hubs,))
         for kd in range(n_hubs):
            dist[kd] = __Distance((kx,ky),(hub_center[kd][0],hub_center[kd][1]), metric=setup['dist_metric'])
         hub = np.argmin(dist)
         g[ky,kx] = hub
   return g





if False:  # examples
  A = ProjectionMatrix()
  Aplus = np.linalg.pinv(A)
  g = CreateImage
  plt.figure(); plt.imshow(g,origin='lower',cmap='seismic'); plt.title('original image');plt.colorbar();
  plt.figure(); plt.imshow((Aplus@A@g.reshape((4096,))).reshape((64,64)),origin='lower',cmap='seismic'); plt.colorbar();

  y = A@g.reshape((4096,))  # projection data
  ghat = Aplus@y
  yhat = A@ghat
  plt.figure(); plt.plot(y,'ko',yhat,'rx');
