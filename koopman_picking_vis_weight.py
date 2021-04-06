#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:08:06 2020

@author: bzfsechi
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import svd, pinv, logm, eig
from sklearn.cluster import KMeans
import cmdtools
from tools import plot_spectrum_strx, Koopman, stroboscopic_inds, hard_chi, nn_weighted
import cmdtools.estimation.voronoi as voronoi
from cmdtools import utils
from cmdtools.estimation.newton_generator import Newton_N
from sklearn.neighbors import NearestNeighbors
import networkx
from check_commutator import check_commutator
from reduction_projection import proj, rebinding
#%%

# data_1 = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt').T
data_1 = np.loadtxt("br_py2_exec400.txt").T
#%%
spectrum_1 = data_1[1:, 50:]
ts1 = data_1[0,50:]
aaa = stroboscopic_inds(ts1)
wl = data_1[1:,0]
#%%
#infgen
nclus = 5
jumps = 2
nstates = 50
spectrum_infgen, picked_inds,centers, K_tens, indices, distances = Koopman(spectrum_1.T, ts1, w=10**7/wl, nstates=nstates, picked_weights=True)

#%%
K = K_tens[1]
eig_k = np.sort(np.linalg.eigvals(K))
eigvec_k = np.linalg.eig(K)[1]
print(eig_k)
#%%
chi_k = cmdtools.analysis.pcca.pcca(K,nclus, pi="uniform")
chi_k_hard = hard_chi(chi_k)
plt.imshow(K)

plt.show()
plt.plot(eig_k, "-o")
plt.show()
#%%
plt.imshow(chi_k_hard, aspect="auto")
plt.yticks(ticks=[])
plt.title("$\chi$")
plt.ylabel("cells")
plt.colorbar()
plt.show()
     #%%
K_c = pinv(chi_k).dot(K.dot(chi_k))
K_c1 = pinv(chi_k.T.dot(chi_k)).dot(chi_k.T.dot(K.dot(chi_k)))
K_c2 = proj(K,nclus, pi="statdistr")
S_c, detSc = rebinding(K, nclus=nclus)
T_c = S_c.dot(K_c) 

#reb = rebinding(K, nclus, pi="statdistr")
#%%
#     print(np.sum(K_c[i], axis =1))
K_c_hard =  chi_k_hard.T.dot(K.dot(chi_k_hard))
plt.imshow(K_c)
plt.colorbar()
#%%
color_list = ["r", "deepskyblue", "fuchsia", "gold","darkgreen","coral","black"]
plot_spectrum_strx(spectrum_1.T,wl, ts1)
for i in range(len(picked_inds)):
    plt.axhline(y=picked_inds[i], color=color_list[np.argmax((chi_k)[i,:])])
plt.show()
#%%
# Infgen = Newton_N(K_tens[:3], 1, 0)
# eig_infgen =  np.sort(np.linalg.eigvals(Infgen))
# chi_infgen = cmdtools.analysis.pcca.pcca(Infgen,nclus)
# plot_spectrum_strx(spectrum_1.T,wl, ts1)
# for i in range(len(picked_inds)):
#     plt.axhline(y=picked_inds[i], color=color_list[np.argmax((chi_infgen)[i,:])])
# # plt.savefig("pcca_nt_br_al_corr_vis.pdf")
# plt.show()

#%%
# Infgen_c = pinv(chi_infgen).dot(Infgen.dot(chi_infgen))
# #reb_q = rebinding(Infgen, nclus, pi="uniform")
# Infgen_c_hard = pinv(hard_chi(chi_infgen)).dot(Infgen.dot(hard_chi(chi_infgen)))
# #print(1/Infgen_c_hard.diagonal(), 1/logm(K_c_hard).diagonal(),1/(K_c_hard-np.ones(K_c.shape[0])).diagonal())
# print("soft", 1/Infgen_c.diagonal(), 1/logm(K_c).diagonal(),1/(K_c-np.ones(K_c.shape[0])).diagonal())
#%%
labels = ["A","B","C","D","E", "F","G"]
for i in range(chi_k.shape[1]):
    plt.plot(ts1[aaa[picked_inds]],chi_k_hard[:,i], "-o", color= color_list[i],label=labels[i])#"$\chi$_%d"%i) 
    
    plt.legend()
    plt.title(r"$\chi_{hard}$ of $K(\tau)$")
    plt.ylabel("concentration")
    plt.xlabel("time/ps")
    
plt.grid()  
plt.xscale("linear")  
#plt.xticks(ticks=aaa[::15])#, labels=(aaa[picked_inds])[::5])

plt.show()
#%%
plt.imshow(spectrum_infgen, cmap="coolwarm", aspect="auto")
plt.xlabel(r"$\lambda$/nm")
plt.ylabel("delay time [ps]")
plt.xticks(np.arange(len(wl), step=60),labels=np.round(data_1[1::60,0]))
        #start with zero but remember to take it off from the lambdas in the data
plt.yticks(np.arange(len(aaa), step=1000),labels=np.round(aaa[::1000],2))
        
plt.colorbar()
plt.show()
#%%
# for i in [0,1,2]:
#     #plt.plot(ts1,Chi[:,i], label="$NMF-\chi$_%d"%i)
#     plt.plot(ts1[aaa[picked_inds]],chi_k[:,i],"-o",color= color_list[i],label="$MSM-\chi$_%d"%i)
#     plt.xlabel("delaytime/ps")
# plt.grid()
# plt.legend()
# plt.show()
#%%
# #dass
DAS = pinv(chi_k).dot(centers)
#%%
plt.figure(figsize=(12,7))
for i in range(chi_k.shape[1]):
    #plt.plot(ts1,Chi[:,i], label="$NMF-\chi$_%d"%i)
    plt.plot(wl,DAS[i,:],"-.",color= color_list[i],label="$MSM-S$_%d"%i)
    plt.xlabel("wavelength $\lambda$/nm")
plt.grid()
plt.legend()
plt.show()
#%%

labels = ["A","B","C","D","E", "F","G"]
plt.figure(figsize=(18,6))
plt.suptitle("$\chi$ and species \n-product ansatz")
plt.subplot(1,2,1)
for i in range(chi_k.shape[1]):
    #plt.plot(ts1,Chi[:,i], label="$NMF-\chi$_%d"%i)
    plt.plot(data_1[95:,0],DAS[i,94:],"-.",color= color_list[i],label="$MSM-S$_%s"%labels[i])
    plt.xlabel("wavelength $\lambda$/nm")
    plt.title("Compounds amplitudes")
plt.grid()
plt.legend()
plt.subplot(1,2,2)

for i in range(chi_k.shape[1]):
    plt.plot(ts1[aaa[picked_inds]],chi_k[:,i], "-o", color= color_list[i],label=labels[i])#"$\chi$_%d"%i) 
    
    plt.legend()
    plt.title(r"$\chi$ of $K(\tau)$")
    plt.ylabel("concentration")
    plt.xlabel("time/ps")
    
plt.grid()  
plt.xscale("linear")  
#plt.xticks(ticks=aaa[::15])#, labels=(aaa[picked_inds])[::5])



plt.show()
#%%
K_c_hard =  pinv(chi_k_hard).dot(K.dot(chi_k_hard))#/ (pinv(chi_k).dot(chi_k)))
#     print(np.sum(K_c[i], axis =1))
plt.imshow(K_c_hard)
plt.colorbar()
plt.show()
#%%
# K_c_graph_soft = networkx.from_numpy_matrix(K_c)
# networkx.draw(K_c_graph_soft, with_labels=True, font_weight='bold')
# plt.show()
#%%
check_commutator(K,nclus=5)
#%%
#ts_new = ts[strobox]
#step_ = int(len(ts_new)/10)
plt.figure(figsize=(7,6))
plt.imshow(spectrum_infgen, cmap="coolwarm",aspect = "auto", alpha=0.8)
plt.colorbar()
plt.title("Pump-probe spectrum of brominated \n aluminium corrole exec.400nm")
plt.xlabel("$\lambda$ [nm]")
plt.ylabel("delay time [ps]")
plt.xticks(np.arange(len(wl), step=60),labels=np.round(wl[1::60]))
#for i in range(len(picked_inds)):
 #   plt.axhline(y=picked_inds[i], color=color_list[np.argmax((chi_k)[i,:])])
#start with zero but remember to take it off from the lambdas in the data
plt.yticks([50,100,150,200,250])
