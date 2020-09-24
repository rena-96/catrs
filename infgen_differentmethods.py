#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:08:06 2020

@author: bzfsechi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, pinv
import cmdtools
#from tools import norm_rows, avg_spectrum
from tools import voronoi_propagator
import cmdtools.estimation.voronoi as voronoi
from cmdtools import utils
from cmdtools.estimation.picking_algorithm import picking_algorithm
#%%
#trialmtx = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt')[1:,1:]
data = np.loadtxt("br_py2_exec400.txt")
#data = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt')
spectrum = data[40:, 1:]
#%%
picked_ind = np.sort(picking_algorithm(spectrum, 29)[1])
center_type = ["kmeans", spectrum[picked_ind,:], spectrum[::5,:]]
list_Koopman_gen = []
for types in center_type:
    #K = voronoi.VoronoiTrajectory(spectrum, 30, centers=types).propagator() 
    K = voronoi_propagator(spectrum,types,29,dt=[1])
    list_Koopman_gen.append(K-np.eye(K.shape[0]))
list_ew = []
for i in range(3):
    list_ew.append(np.sort(np.linalg.eigvals(list_Koopman_gen[i])))
list_Chi = []
for c in range(3):
    list_Chi.append(cmdtools.analysis.pcca.pcca(list_Koopman_gen[c],5))
    #%%
for j in range(3):
    plt.imshow(list_Koopman_gen[j])
    plt.show()
    plt.plot(list_ew[j], "-o")
    plt.show()
    plt.imshow(list_Chi[j], aspect="auto")
    plt.show()
    #%%
Q_c = []
for i in range(3):
    Q_c.append( pinv(list_Chi[i]).dot(list_Koopman_gen[i].dot(list_Chi[i])))#/ (pinv(list_Chi[i]).dot(list_Chi[i])))
    print(np.sum(Q_c[i], axis =1))
#plt.imshow(Q_c)
   #%%
color_list = ["lime", "ivory", "deepskyblue", "red", "gold"]
plt.figure(figsize=(18,6))
plt.subplot(1, 3, 1)
plt.imshow(spectrum, cmap='inferno', aspect='auto')
plt.colorbar()
plt.title("Kmeans")
plt.xticks(np.arange(len(data[0,1:]), step=60),labels=np.round(data[0,1::60]))
plt.yticks(np.arange(len(data[40:,0]), step=20),labels=np.round(data[40::20,0],2))
#for  -
plt.subplot(1, 3, 2)
plt.imshow(spectrum, cmap="inferno",aspect = "auto")
plt.colorbar()
plt.title("Picking algorithm")
plt.xticks(np.arange(len(data[0,1:]), step=60),labels=np.round(data[0,1::60]))
plt.yticks(np.arange(len(data[40:,0]), step=20),labels=np.round(data[40::20,0],2))
for i in range(len(picked_ind)):
    plt.axhline(y=picked_ind[i], color=color_list[np.argmax((list_Chi[1])[i,:])])
plt.subplot(1, 3, 3)
plt.imshow(spectrum, cmap="inferno",aspect = "auto")
plt.colorbar()
plt.title("Regular grid")
plt.xticks(np.arange(len(data[0,1:]), step=60),labels=np.round(data[0,1::60]))
plt.yticks(np.arange(len(data[40:,0]), step=20),labels=np.round(data[40::20,0],2))
for j in np.arange(spectrum.shape[0], step=5):
    plt.axhline(y=j, color=color_list[np.argmax((list_Chi[2])[int(j/5),:])])
plt.show()
#%%
eigenvals = np.linalg.eigvals(Q_c)
print(np.sort(1/eigenvals))