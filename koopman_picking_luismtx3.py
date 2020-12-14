#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:08:06 2020

@author: bzfsechi
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import svd, pinv, logm
from sklearn.cluster import KMeans
import cmdtools
from tools import voronoi_koopman_picking, plot_spectrum_strx, stroboscopic_inds
import cmdtools.estimation.voronoi as voronoi
from cmdtools import utils
from cmdtools.estimation.picking_algorithm import picking_algorithm
from cmdtools.estimation.newton_generator import Newton_N
from sklearn.neighbors import NearestNeighbors
#%%

data_1 = np.loadtxt("matrix_2.dat")
# data_2 = np.loadtxt("matrix_2.dat")
# data_3 = np.loadtxt("matrix_3.dat")
#data = np.loadtxt('iso_br_al_cor_py2_420nm_ex_ir.txt')
#%%
spectrum_1 = data_1[1:, 50:]
ts1 = data_1[0,50:]
aaa = stroboscopic_inds(ts1)

#%%
# K, spectrum_new, picked_inds = voronoi_koopman_picking(spectrum_1.T,20,timeseries=data_1[0,102:],dt=1)
#%%
#infgen
nclus = 2
jumps = 10
nstates = 50
strobox = stroboscopic_inds(ts1)

spectrum_infgen = (spectrum_1.T)[strobox,:]
K_tens = np.zeros((jumps,nstates, nstates))

picked_inds = np.sort(picking_algorithm(spectrum_infgen,nstates)[1])
centers = spectrum_infgen[picked_inds,:]
inds =  (NearestNeighbors()
          .fit(centers).kneighbors(spectrum_infgen, 1, False)
          .reshape(-1))
#tau=1
# # print(inds, "inds of K")
for j in range(jumps):
    
    for i in range(0,len(inds)-j):
        (K_tens[j])[inds[i], inds[i+j]] += 1
    K_tens[j] = utils.rowstochastic(K_tens[j])
#%%
K = K_tens[1]
eig_k = np.sort(np.linalg.eigvals(K))
eigvec_k = np.linalg.eig(K)[1]
print(eig_k)
#%%
chi_k = cmdtools.analysis.pcca.pcca(K,nclus)

plt.imshow(K)
plt.show()
plt.plot(eig_k, "-o")
plt.show()
plt.imshow(chi_k, aspect="auto")
plt.show()
     #%%
K_c =  pinv(chi_k).dot(K.dot(chi_k))#/ (pinv(chi_k).dot(chi_k)))
#     print(np.sum(K_c[i], axis =1))
plt.imshow(K_c)
plt.colorbar()
#     #%%
color_list = ["g", "ivory", "deepskyblue", "fuchsia", "gold","darkgreen","coral"]
plot_spectrum_strx(spectrum_1.T,data_1[1:,0], ts1)
for i in range(len(picked_inds)):
    plt.axhline(y=picked_inds[i], color=color_list[np.argmax((chi_k)[i,:])])
plt.show()
#%%
Infgen = Newton_N(K_tens[:4], 1, 0)
eig_infgen =  np.sort(np.linalg.eigvals(Infgen))
chi_infgen = cmdtools.analysis.pcca.pcca(Infgen,nclus)
color_list = ["g", "ivory", "deepskyblue", "fuchsia", "gold","darkgreen","coral"]
plot_spectrum_strx(spectrum_1.T,data_1[1:,0], ts1)
for i in range(len(picked_inds)):
    plt.axhline(y=picked_inds[i], color=color_list[np.argmax((chi_infgen)[i,:])])
# plt.savefig("pcca_nt_br_al_corr_vis.pdf")
plt.show()

#%%
Infgen_c = pinv(chi_infgen).dot(Infgen.dot(chi_infgen))

print(1/Infgen_c.diagonal(), 1/logm(K_c).diagonal(),1/(K_c-np.ones(K_c.shape[0])).diagonal())
#%%
for i in range(chi_k.shape[1]):
    plt.plot(aaa[picked_inds],chi_k[:,i], "-o", label="$\chi$_%d"%i) 
    plt.grid()
    plt.legend()
    plt.title("$\chi$")
    plt.ylabel("concentration")
    plt.xlabel("time/ps")
#    plt.xticks(ticks=np.arange(len(aaa), step=1000),labels=aaa[::1000])
plt.show()
#%%
plt.imshow(spectrum_infgen, cmap="coolwarm", aspect="auto")
plt.xlabel(r"$\nu/10^3$cm-1")
plt.ylabel("delay time [ps]")
plt.xticks(np.arange(len(data_1[1:,0]), step=150),labels=np.round(data_1[1::150,0]/1000))
        #start with zero but remember to take it off from the lambdas in the data
plt.yticks(np.arange(len(aaa), step=1000),labels=np.round(aaa[::1000],2))
        
plt.colorbar()
#%%
for i in [0,1]:
    plt.plot(ts1,Chi[:,i], label="$NMF-\chi$_%d"%i)
    # plt.plot(aaa[picked_inds],chi_k[:,i],"-o",label="$MSM-\chi$_%d"%i)
    plt.xlabel("delaytime/ps")
    plt.legend()