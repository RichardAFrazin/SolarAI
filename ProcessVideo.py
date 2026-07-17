#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:52:07 2026

@author: rfrazin
"""

import numpy as np
import cv2

mp4file = "highway.mp4"; cen1 = (200,381); cen2 = (200, 262); npix=80
setupdict = {'sz': npix, 'n_ang': 26}


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



def rebin2Darray(array, new_shape): #rebin by averaging
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
