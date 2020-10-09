#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:52:52 2020

@author: bzfsechi
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, pinv
from sklearn.cluster import KMeans
import cmdtools
from tools import voronoi_propagator, analyse_spectrum_picking_alg
import cmdtools.estimation.voronoi as voronoi
from cmdtools import utils
from cmdtools.estimation.picking_algorithm import picking_algorithm
#%%

spectrum_1 = np.loadtxt("matrix_1.dat").T
#spectrum_2 = np.loadtxt("matrix_2.dat")[1:,1:].T
#spectrum_3 = np.loadtxt("matrix_3.dat")[1:,1:].T
time = spectrum_1[1:,0]
freq = spectrum_1[0,1:]
spectrum_1 = spectrum_1[1:,1:]
#%%

    #%%
K_c1 = analyse_spectrum_picking_alg(spectrum_1, time, 20)
#%%
timeconst = 1/np.log(np.linalg.eigvals(K_c1[0]))
#%%
plt.figure(figsize=(9,9))
color_list = ["g", "ivory", "deepskyblue", "fuchsia", "gold", "darkorchid", "seashell", "lime", "orange"]
plt.imshow(spectrum_1, cmap="inferno",aspect = "auto")
plt.colorbar()
plt.title("Picking algorithm")
#plt.xticks(np.arange(len(data[0,1:196]), step=60),labels=np.round(data[0,1:196:50]))
#plt.yticks(np.arange(len(data[40:,0]), step=20),labels=np.round(data[40::20,0],2))
for i in range(len(K_c1[2])):
    plt.axhline(y=(K_c1[2])[i], color=color_list[np.argmax((K_c1[1])[i,:])])
    #%%
for i in range(np.shape(K_c1[1])[1]):
    plt.plot((K_c1[1])[:,i])
plt.show()    
#for j in range(3):
#    plt.imshow(list_Koopman[j])
#    plt.show()
#    plt.plot(list_ew[j], "-o")
#    plt.show()
#    plt.imshow(list_Chi[j], aspect="auto")
#    plt.show()
#    #%%
#K_c = []
#for i in range(3):
#    K_c.append( pinv(list_Chi[i]).dot(list_Koopman[i].dot(list_Chi[i])))#/ (pinv(list_Chi[i]).dot(list_Chi[i])))
#    print(np.sum(K_c[i], axis =1))
##plt.imshow(K_c)
#    #%%

#plt.figure(figsize=(18,6))
#plt.subplot(1, 3, 1)
#plt.imshow(spectrum, cmap='inferno', aspect='auto')
#plt.colorbar()
#plt.title("Kmeans")
#plt.xticks(np.arange(len(data[0,1:196]), step=60),labels=np.round(data[0,1:196:50]))
#plt.yticks(np.arange(len(data[40:,0]), step=20),labels=np.round(data[40::20,0],2))
#plt.subplot(1, 3, 2)

#plt.subplot(1, 3, 3)
#plt.imshow(spectrum, cmap="inferno",aspect = "auto")
#plt.colorbar()
#plt.title("Regular grid")
#plt.xticks(np.arange(len(data[0,1:196]), step=60),labels=np.round(data[0,1:196:50]))
#plt.yticks(np.arange(len(data[40:,0]), step=20),labels=np.round(data[40::20,0],2))
#for j in np.arange(spectrum.shape[0]-1, step=7):
#   # print(j)
#    plt.axhline(y=j, color=color_list[np.argmax((list_Chi[2])[int(j/7),:])])
#plt.show()
##%%
##eigenvalsK = np.log(np.real(np.linalg.eigvals(K_c)))
##eigenvalsK = np.linalg.eigvals(K_c)
#print(np.sort(1/eigenvalsK))