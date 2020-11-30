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
from cmdtools.estimation.newton_generator import Newton_N
from sklearn.neighbors import NearestNeighbors
#%%

data = np.loadtxt("br_py2_exec400.txt")
#data = np.loadtxt('iso_br_al_cor_py2_420nm_ex_ir.txt')
spectrum = data[43:, 1:]

#%%
K, spectrum_new, picked_inds = voronoi_koopman_picking(spectrum,17,timeseries=data[43:,0],dt=1)

    #%%
eig_k = np.sort(np.linalg.eigvals(K))
eigvec_k = np.linalg.eig(K)[1]
print(eig_k)
#%%
chi_k = cmdtools.analysis.pcca.pcca(K,5)

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
color_list = ["g", "ivory", "deepskyblue", "fuchsia", "gold","darkgreen","coral"]
plot_spectrum_strx(spectrum,data[0,1:], data[43:,0])
for i in range(len(picked_inds)):
    plt.axhline(y=picked_inds[i], color=color_list[np.argmax((chi_k)[i,:])])
plt.show()
#%%
ts = data[43:,0]
#print(ts[stroboscopic_inds(ts)])
#%%
#infgen
jumps = 4
nstates = 17
strobox = stroboscopic_inds(ts)
spectrum_infgen = spectrum[strobox,:]
K_tens = np.zeros((jumps,nstates, nstates))
picked_inds = np.sort(picking_algorithm(spectrum_infgen,nstates)[1])
centers = spectrum_infgen[picked_inds,:]
inds =  (NearestNeighbors()
         .fit(centers).kneighbors(spectrum_infgen, 1, False)
         .reshape(-1))
#tau=1
# print(inds, "inds of K")
for j in range(jumps):
    
    for i in range(0,len(inds)-j):
       (K_tens[j])[inds[i], inds[i+j]] += 1
    K_tens[j] = utils.rowstochastic(K_tens[j])
#%%
Infgen = Newton_N(K_tens, 1, 0)
eig_infgen =  np.sort(np.linalg.eigvals(Infgen))
chi_infgen = cmdtools.analysis.pcca.pcca(Infgen,5)
color_list = ["g", "ivory", "deepskyblue", "fuchsia", "gold","darkgreen","coral"]
plot_spectrum_strx(spectrum,data[0,1:], data[43:,0])
for i in range(len(picked_inds)):
    plt.axhline(y=picked_inds[i], color=color_list[np.argmax((chi_infgen)[i,:])])
plt.savefig("pcca_nt_br_al_corr_vis.pdf")
plt.show()