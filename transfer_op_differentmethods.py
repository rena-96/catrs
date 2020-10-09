#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:08:06 2020

@author: bzfsechi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, pinv
from sklearn.cluster import KMeans
import cmdtools
from tools import voronoi_propagator
import cmdtools.estimation.voronoi as voronoi
from cmdtools import utils
from cmdtools.estimation.picking_algorithm import picking_algorithm
#%%

data = np.loadtxt("br_py2_exec400.txt")
#data = np.loadtxt('iso_br_al_cor_py2_420nm_ex_ir.txt')
spectrum = data[42:, 1:]
#%%
#def by_kmeans(X, n):
#    k = KMeans(n_clusters=n).fit(X) 
#    kcenters = k.cluster_centers_
##    print(kcenters)
##    inds = []
##    for i in range(len(kcenters)):
##       print(np.where((data[42:,1:]==[kcenters[i]]).all(axis=1)))
##    return np.sort(inds)
#    return(kcenters)
#kmeans_ind = by_kmeans(spectrum, 6)
#%%
picked_ind = np.sort(picking_algorithm(spectrum, 15)[1])
center_type = ["kmeans", spectrum[picked_ind,:], (spectrum[::9,:])[:-1]]
#%%
list_Koopman = []
for types in center_type:
   # K = voronoi.VoronoiTrajectory(spectrum, 29, centers=types).propagator() 
    K = voronoi_propagator(spectrum,types,15,dt=data[1:,0])
    list_Koopman.append(K)
    #%%
list_ew = []
for i in range(3):
    print(np.sort(np.linalg.eigvals(list_Koopman[i])))
    list_ew.append(np.sort(np.linalg.eigvals(list_Koopman[i])))
list_Chi = []
for c in range(3):
    list_Chi.append(cmdtools.analysis.pcca.pcca(list_Koopman[c],5))
    #%%
for j in range(3):
    plt.imshow(list_Koopman[j])
    plt.show()
    plt.plot(list_ew[j], "-o")
    plt.show()
    plt.imshow(list_Chi[j], aspect="auto")
    plt.show()
    #%%
K_c = []
for i in range(3):
    K_c.append( pinv(list_Chi[i]).dot(list_Koopman[i].dot(list_Chi[i])))#/ (pinv(list_Chi[i]).dot(list_Chi[i])))
    print(np.sum(K_c[i], axis =1))
#plt.imshow(K_c)
    #%%
color_list = ["g", "ivory", "deepskyblue", "fuchsia", "gold"]
plt.figure(figsize=(18,6))
plt.subplot(1, 3, 1)
plt.imshow(spectrum, cmap='inferno', aspect='auto')
plt.colorbar()
plt.title("Kmeans")
plt.xticks(np.arange(len(data[0,1:]), step=60),labels=np.round(data[0,1::60]))
plt.yticks(np.arange(len(data[42:,0]), step=20),labels=np.round(data[42::20,0],2))
plt.subplot(1, 3, 2)
plt.imshow(spectrum, cmap="inferno",aspect = "auto")
plt.colorbar()
plt.title("Picking algorithm")
plt.xticks(np.arange(len(data[0,1:]), step=60),labels=np.round(data[0,1::60]))
plt.yticks(np.arange(len(data[42:,0]), step=20),labels=np.round(data[42::20,0],2))
for i in range(len(picked_ind)):
    plt.axhline(y=picked_ind[i], color=color_list[np.argmax((list_Chi[1])[i,:])])
plt.subplot(1, 3, 3)
plt.imshow(spectrum, cmap="inferno",aspect = "auto")
plt.colorbar()
plt.title("Regular grid")
plt.xticks(np.arange(len(data[0,1:]), step=60),labels=np.round(data[0,1::60]))
plt.yticks(np.arange(len(data[42:,0]), step=20),labels=np.round(data[42::20,0],2))
for j in np.arange(spectrum.shape[0], step=10):
   # print(j)
    plt.axhline(y=j, color=color_list[np.argmax((list_Chi[2])[int(j/10),:])])
plt.show()
#%%
eigenvalsK = np.log(np.real(np.linalg.eigvals(K_c)))
print(np.sort(1/eigenvalsK))