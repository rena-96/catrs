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
from tools import voronoi_koopman_picking, plot_spectrum_strx, stroboscopic_inds
import cmdtools.estimation.voronoi as voronoi
from cmdtools import utils
from cmdtools.estimation.picking_algorithm import picking_algorithm
#%%

data = np.loadtxt("br_py2_exec400.txt")
#data = np.loadtxt('iso_br_al_cor_py2_420nm_ex_ir.txt')
spectrum = data[43:, 1:]

#%%
K, spectrum_new, picked_inds = voronoi_koopman_picking(spectrum,30,timeseries=data[43:,0],dt=1)

    #%%
eig_k = np.sort(np.linalg.eigvals(K))
print(eig_k)
chi_k = cmdtools.analysis.pcca.pcca(K,4)

plt.imshow(K)
plt.show()
plt.plot(eig_k, "-o")
plt.show()
plt.imshow(chi_k, aspect="auto")
plt.show()
#     #%%
K_c =  pinv(chi_k).dot(K.dot(chi_k))#/ (pinv(chi_k).dot(chi_k)))
#     print(np.sum(K_c[i], axis =1))
plt.imshow(K_c)
plt.colorbar()
    #%%
color_list = ["g", "ivory", "deepskyblue", "fuchsia", "gold"]
plt.figure(figsize=(6,5))

plt.imshow(spectrum_new, cmap="inferno",aspect = "auto")
plt.colorbar()
plt.title("Picking algorithm")
plt.xticks(np.arange(len(data[0,1:]), step=60),labels=np.round(data[0,1::60]))
plt.yticks(np.arange(len(data[42:,0]), step=20),labels=np.round(data[42::20,0],2))
for i in range(len(picked_inds)):
    plt.axhline(y=picked_inds[i], color=color_list[np.argmax((chi_k)[i,:])])
plt.show()
# #%%
# eigenvalsK = np.log(np.real(np.linalg.eigvals(K_c)))
# print(np.sort(1/eigenvalsK))
#%%
plot_spectrum_strx(spectrum,data[0,1:], data[43:,0])
for i in range(len(picked_inds)):
    plt.axhline(y=picked_inds[i], color=color_list[np.argmax((chi_k)[i,:])])
plt.show()
