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
from infgen_4ways import infgen_3ways
#%%

# data_1 = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt').T
data_1 = np.loadtxt("br_py2_exec400.txt").T
# <<<<<<< HEAD
# #%%#start 500 ps
# spectrum_1 = data_1[1:, 45:]
# ts1 = data_1[0,45:]
# =======
#%%#start 500 ps 146
spectrum_1 = data_1[1:, 45:]
ts1 = data_1[0,45:]
# >>>>>>> ab764473d418df46488b3bc0f92a3317481dc215
aaa = stroboscopic_inds(ts1)
wl = data_1[1:,0]
#%%
#infgen
nclus = 5
jumps = 4
nstates = 50
spectrum_infgen, picked_inds,centers, K_tens, indices, distances = Koopman(spectrum_1.T, ts1, w=10**7/wl, nstates=nstates, jumps=jumps, picked_weights=True)

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
K_c2 = proj(K,nclus, pi="uniform")
S_c, detSc = rebinding(K, nclus=nclus)
T_c = S_c.dot(K_c) 


#%%
color_list = ["r", "deepskyblue", "fuchsia", "gold","darkgreen","coral","black"]
cmap = plt.get_cmap("tab10")
plot_spectrum_strx(spectrum_1.T,wl, ts1)
for i in range(len(picked_inds)):
    plt.axhline(y=picked_inds[i], color=cmap(np.argmax((chi_k)[i,:])))
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

labels = ["$b_1$","$b_2$","$b_3$","$b_4$","$b_5$", "$B_6$","G"]
#labels = ["A","B","C","D","E", "F","G"]
plt.figure(figsize=(18,6))
plt.suptitle("$\chi$ and species \n-product ansatz")
plt.subplot(1,2,2)
for i in range(chi_k.shape[1]):
    #plt.plot(ts1,Chi[:,i], label="$NMF-\chi$_%d"%i)
    plt.scatter(data_1[95:,0],DAS[i,94:],marker= ".",color= cmap(i),label=labels[i])
    plt.xlabel("wavelength $\lambda$/nm")
    plt.title("Compounds amplitudes")
    plt.ylabel("$\Delta A$")
plt.grid()
plt.legend()
plt.subplot(1,2,1)

for i in range(chi_k.shape[1]):
    plt.plot(ts1[aaa[picked_inds]],chi_k[:,i], "-o", color= cmap(i),label=labels[i])#"$\chi$_%d"%i) 
    

    plt.title(r"$\chi$ of $K(\tau)$")
    plt.ylabel("concentration")
    plt.xlabel("time/ps")
plt.legend(ncol=5)
plt.grid()  
plt.xscale("linear")  
#plt.xticks(ticks=aaa[::15])#, labels=(aaa[picked_inds])[::5])


#plt.savefig("br-corrole-20vor-weighted-35ps.pdf")
#>>>>>>> ab764473d418df46488b3bc0f92a3317481dc215

plt.show()
#%%
# K_c_hard =  pinv(chi_k_hard).dot(K.dot(chi_k_hard))#/ (pinv(chi_k).dot(chi_k)))
# #     print(np.sum(K_c[i], axis =1))
plt.imshow(K)
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
# for i in range(len(picked_inds)):
#      plt.axhline(y=picked_inds[i], color=cmap(np.argmax((chi_k)[i,:])))
# #start with zero but remember to take it off from the lambdas in the data
plt.yticks([50,100,150,200,250])
plt.legend()
plt.savefig("br-spectrum.pdf")
#%%
#transform K tens with pcca+
K_pcca = np.zeros((jumps,nclus,nclus))
for i in range(jumps):
    print(i)
    K_pcca[i] = proj(K_tens[i],nclus, pi="uniform")
 #%%   
infgen = infgen_3ways(K_pcca )
taus = []
for j in range(3):
    taus.append(1/infgen[j].diagonal())
#%%
# manyfigs, axs = plt.subplots(1,4)
# #the spectrum
# axs[0].imshow(spectrum_infgen, cmap="coolwarm",aspect = "auto", alpha=0.8)
# #plt.colorbar(ax=axs[0])
# axs[0].set_title("Pump-probe spectrum of brominated \n aluminium corrole exec.400nm")
# axs[0].set_xlabel("$\lambda$ [nm]")
# axs[0].set_ylabel("delay time [ps]")
# axs[0].set_xticks(np.arange(len(wl), step=60))#,label=np.round(wl[1::60]))
# #for i in range(len(picked_inds)):
#   #   plt.axhline(y=picked_inds[i], color=color_list[np.argmax((chi_k)[i,:])])
# #start with zero but remember to take it off from the lambdas in the data
# axs[0].set_yticks([50,100,150,200,250])

# # the spectrum w the VOronoi cells
# axs[1].imshow(spectrum_infgen, cmap="coolwarm",aspect = "auto", alpha=0.8)
# #plt.colorbar()
# axs[1].set_title("Pump-probe spectrum of brominated \n aluminium corrole exec.400nm")
# axs[1].set_xlabel("$\lambda$ [nm]")
# axs[1].set_ylabel("delay time [ps]")
# axs[1].set_xticks(np.arange(len(wl), step=60),label=np.round(wl[1::60]))
# # for i in range(len(picked_inds)):
     # plt.axhline(y=picked_inds[i], color="black")
#start with zero but remember to take it off from the lambdas in the data
# axs[1].set_yticks([50,100,150,200,250])
for i in range(chi_k.shape[1]):
    #plt.plot(ts1,Chi[:,i], label="$NMF-\chi$_%d"%i)The
    plt.scatter(data_1[95:,0],DAS[i,94:],marker= ".",color= cmap(i),label=labels[i])
    plt.xlabel("wavelength $\lambda$/nm")
    plt.title("Compounds amplitudes")
    plt.ylabel("$\Delta A$")
    plt.grid()
    plt.legend()
    plt.show()