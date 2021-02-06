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
from tools import plot_spectrum_strx, Koopman, stroboscopic_inds, hard_chi
import cmdtools.estimation.voronoi as voronoi
from cmdtools import utils
from cmdtools.estimation.picking_algorithm import picking_algorithm
from cmdtools.estimation.newton_generator import Newton_N
from sklearn.neighbors import NearestNeighbors
import networkx
from check_commutator import check_commutator
#%%

# data_1 = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt').T
data_1 = np.loadtxt("br_py2_exec400.txt").T
#%%
spectrum_1 = data_1[1:, 50:]
ts1 = data_1[0,50:]
aaa = stroboscopic_inds(ts1)


#%%
#infgen
nclus = 5
jumps = 2
nstates = 20

spectrum_infgen, picked_inds,centers, K_tens = Koopman(spectrum_1, ts1)

#%%
K = K_tens[1]
eig_k = np.sort(np.linalg.eigvals(K))
eigvec_k = np.linalg.eig(K)[1]
print(eig_k)
#%%
chi_k = cmdtools.analysis.pcca.pcca(K,nclus)
chi_k_hard = hard_chi(chi_k)
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
color_list = ["r", "deepskyblue", "fuchsia", "gold","darkgreen","coral","black"]
plot_spectrum_strx(spectrum_1.T,data_1[1:,0], ts1)
for i in range(len(picked_inds)):
    plt.axhline(y=picked_inds[i], color=color_list[np.argmax((chi_k)[i,:])])
plt.show()
#%%
Infgen = Newton_N(K_tens[:4], 1, 0)
eig_infgen =  np.sort(np.linalg.eigvals(Infgen))
chi_infgen = cmdtools.analysis.pcca.pcca(Infgen,nclus)
plot_spectrum_strx(spectrum_1.T,data_1[1:,0], ts1)
for i in range(len(picked_inds)):
    plt.axhline(y=picked_inds[i], color=color_list[np.argmax((chi_infgen)[i,:])])
# plt.savefig("pcca_nt_br_al_corr_vis.pdf")
plt.show()

#%%
Infgen_c = pinv(chi_infgen).dot(Infgen.dot(chi_infgen))

print(1/Infgen_c.diagonal(), 1/logm(K_c).diagonal(),1/(K_c-np.ones(K_c.shape[0])).diagonal())
#%%
labels = ["A","B","C","D","E"]
for i in range(chi_k.shape[1]):
    plt.plot(ts1[aaa[picked_inds]],chi_k_hard[:,i], "-o", color= color_list[i],label=labels[i])#"$\chi$_%d"%i) 
    
    plt.legend()
    plt.title(r"$\chi$ of $K(\tau)$")
    plt.ylabel("concentration")
    plt.xlabel("time/ps")
    
plt.grid()  
plt.xscale("linear")  
#plt.xticks(ticks=aaa[::15])#, labels=(aaa[picked_inds])[::5])

plt.show()
#%%
plt.imshow(spectrum_infgen, cmap="coolwarm", aspect="auto")
plt.xlabel(r"$\nu/10^3$cm-1")
plt.ylabel("delay time [ps]")
plt.xticks(np.arange(len(data_1[1:,0]), step=150),labels=np.round(data_1[1::150,0]/1000))
        #start with zero but remember to take it off from the lambdas in the data
plt.yticks(np.arange(len(aaa), step=1000),labels=np.round(aaa[::1000],2))
        
plt.colorbar()
plt.show()
#%%
for i in [0,1,2]:
    #plt.plot(ts1,Chi[:,i], label="$NMF-\chi$_%d"%i)
    plt.plot(ts1[aaa[picked_inds]],chi_k[:,i],"-o",color= color_list[i],label="$MSM-\chi$_%d"%i)
    plt.xlabel("delaytime/ps")
plt.grid()
plt.legend()
plt.show()
#%%
# #dass
DAS = pinv(chi_k).dot(centers)
#%%
plt.figure(figsize=(12,7))
for i in range(chi_k.shape[1]):
    #plt.plot(ts1,Chi[:,i], label="$NMF-\chi$_%d"%i)
    plt.plot(data_1[1:,0],DAS[i,:],"-.",color= color_list[i],label="$MSM-S$_%d"%i)
    plt.xlabel("wavelength $\lambda$/nm")
plt.grid()
plt.legend()
plt.show()
#%%
plt.figure(figsize=(12,7))
for i in range(chi_k.shape[1]):
    #plt.plot(ts1,Chi[:,i], label="$NMF-\chi$_%d"%i)
    plt.plot(data_1[95:,0],DAS[i,94:],"-.",color= color_list[i],label="$MSM-S$_%d"%i)
    plt.xlabel("wavelength $\lambda$/nm")
plt.grid()
plt.legend()
plt.show()
#%%
K_c_hard =  pinv(chi_k_hard).dot(K.dot(chi_k_hard))#/ (pinv(chi_k).dot(chi_k)))
#     print(np.sum(K_c[i], axis =1))
plt.imshow(K_c_hard)
plt.colorbar()
plt.show()
#%%
K_c_graph_soft = networkx.from_numpy_matrix(K_c)
networkx.draw(K_c_graph_soft, with_labels=True, font_weight='bold')
plt.show()
#%%
check_commutator(K,nclus=5)