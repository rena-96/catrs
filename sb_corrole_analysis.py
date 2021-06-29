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
from tools import plot_spectrum_strx, Koopman, stroboscopic_inds, hard_chi, nn_weighted
import cmdtools.estimation.voronoi as voronoi
from cmdtools import utils
from cmdtools.estimation.picking_algorithm import picking_algorithm
from cmdtools.estimation.newton_generator import Newton_N
from sklearn.neighbors import NearestNeighbors
import networkx
from check_commutator import check_commutator
from reduction_projection import rebinding, proj
from infgen_4ways import infgen_3ways
#%%


# data_1 = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt').T
file = np.load("SB_corrole_400nm.npz")
data_1 = file["data"][0,:,:,0,:]
wl = file["wl"][:,0]
ts = file["t"]

#%%
spectrum_1 = np.nanmean(data_1, axis=2)
#start analysis at 300 fs and 300 ps  and  400-700 nm circa (198+44)
ts = ts[46:(198+44)]*.001
wl = wl[500:1448]
spectrum_1 = spectrum_1[46:(198+44),500:1448]
aaa = stroboscopic_inds(ts)

#%%
#infgen
nclus = 4
jumps = 2
nstates = 30
spectrum_infgen, picked_inds,centers, K_tens, indices, distances = Koopman(spectrum_1, ts,jumps=jumps, nstates=nstates, w=10**7/wl, picked_weights=True)
#%%
plt.imshow(spectrum_infgen, cmap="coolwarm", aspect="auto")
plt.xlabel(r"$\lambda/$nm")
plt.ylabel("delay time [ps]")
plt.xticks(np.arange(len(wl), step=120),labels=np.round(wl[1::120]))
#plt.xticks(np.arange(wl[0],wl[-1], step=-120))
        #start with zero but remember to take it off from the lambdas in the data
#plt.yticks([0,10,20,30,40,50])#,100,150,200,250])
        
plt.colorbar()
#plt.savefig("sb_spectrum_250ps.svg")
plt.show()
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
K_c =  pinv(chi_k).dot(K.dot(chi_k))

   #%%
plt.figure(figsize=(13,12))
cmap_states = plt.get_cmap("tab10")
plt.title("Sb-Corrole pump-probe specturm \n ex 400nm")#" \n Assignment of dominant conformaiton from PCCA+ \n %d Voronoi cells"%nstates)
plt.imshow(spectrum_infgen,cmap="coolwarm", aspect="auto")
plt.xlabel(r"$\lambda/$nm")
plt.ylabel("delay time [ps]")
plt.xticks(np.arange(len(wl), step=120),labels=np.round(wl[1::120]))

#plt.xticks(np.arange(wl[0],wl[-1], step=-120))
        #start with zero but remember to take it off from the lambdas in the data
plt.yticks(np.arange(0,1200, step=50))
#plt.yticks([0,10,20,30,40,50])       
plt.colorbar()
for i in range(len(picked_inds)):
    plt.axhline(y=picked_inds[i], color=cmap_states(np.argmax((chi_k)[i,:])))
#lt.savefig("sb_corrole_chi_50vor_250ps.pdf")
plt.show()
#%%
# K_c_hard =  pinv(chi_k_hard).dot(K.dot(chi_k_hard))
#%%
#Infgen = Newton_N(K_tens[:3], 1, 0)
#eig_infgen =  np.sort(np.linalg.eigvals(Infgen))
#chi_infgen = cmdtools.analysis.pcca.pcca(Infgen,nclus)
# plot_spectrum_strx(spectrum_1,wl, ts)
# for i in range(len(picked_inds)):
#     plt.axhline(y=picked_inds[i], color=color_list[np.argmax((chi_infgen)[i,:])])
# # plt.savefig("pcca_nt_br_al_corr_vis.pdf")
# plt.show()

#%%
# Infgen_c = pinv(chi_infgen).dot(Infgen.dot(chi_infgen))
# Infgen_c_hard = pinv(hard_chi(chi_infgen)).dot(Infgen.dot(hard_chi(chi_infgen)))
# print(1/Infgen_c_hard.diagonal(), 1/logm(K_c_hard).diagonal(),1/(K_c_hard-np.ones(K_c.shape[0])).diagonal())
#%%
labels = ["$D_1$","$D_2$","$D_3$","$D_4$","$D_5$", "F", "G", "H´"]
for i in range(chi_k.shape[1]):
    plt.plot(ts[aaa[picked_inds]],chi_k_hard[:,i], "-o", color= cmap_states(i),label=labels[i])#"$\chi$_%d"%i) 
    
    plt.legend()
    plt.title(r"$\chi$ of $K(\tau)$")
    plt.ylabel("concentration")
    plt.xlabel("time/ps")
    
plt.grid()  
plt.xscale("linear") 
# plt.xticks(ticks=np.round(ts[aaa[picked_inds]]) )
#plt.xticks(ticks=aaa[::15])#, labels=(aaa[picked_inds])[::5])

plt.show()
#%%
# plt.imshow(spectrum_infgen, cmap="coolwarm", aspect="auto")
# plt.xlabel(r"$\nu/10^3$cm-1")
# plt.ylabel("delay time [ps]")
# # plt.xticks(np.arange(len(wl), step=120),labels=np.round(wl))
# # plt.xticks(np.round(wl[0::100]))
#         #start with zero but remember to take it off from the lambdas in the data
# plt.yticks(np.arange(len(aaa), step=1000),labels=np.round(aaa[::1000],2))
        
# plt.colorbar()
# plt.show()
#%%
# for i in [0,1,2]:
#     #plt.plot(ts,Chi[:,i], label="$NMF-\chi$_%d"%i)
#     plt.plot(ts[aaa[picked_inds]],chi_k[:,i],"-o", color= cmap_states(i),label="$MSM-\chi$_%d"%i)
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
    plt.figure(figsize=(12,7))
    #plt.plot(ts,Chi[:,i], label="$NMF-\chi$_%d"%i)
    plt.plot(wl,DAS[i,:],"-.",color=cmap_states(i),label="$MSM-S$_%d"%i)
    plt.xlabel("wavelength $\lambda$/nm")
    plt.grid()
    plt.legend()
    plt.show()
#%%
labels = ["$d_1$","$d_2$","$d_3$","$d_4$","E", "F","G"]
plt.figure(figsize=(18,6))
plt.suptitle("$\chi$ and species \n-product ansatz")
plt.subplot(1,2,2)
for i in range(chi_k.shape[1]):
    #plt.plot(ts1,Chi[:,i], label="$NMF-\chi$_%d"%i)
    plt.plot(wl,DAS[i,:],"-.",color= cmap_states(i),label=labels[i])
    plt.xlabel("wavelength $\lambda$/nm")
    plt.ylabel("$\Delta A$")
    plt.title("Compounds amplitudes")
plt.grid()
plt.legend()
plt.subplot(1,2,1)

for i in range(chi_k.shape[1]):
    plt.plot(ts[aaa[picked_inds]],chi_k[:,i], "-o", color= cmap_states(i),label=labels[i])#"$\chi$_%d"%i) 
    
    plt.legend()
    plt.title(r"$\chi$ of $K(\tau)$")
    plt.ylabel("concentration")
    plt.xlabel("time/ps")
    
plt.grid()  
plt.xscale("linear")  
#plt.xticks(ticks=aaa[::15])#, labels=(aaa[picked_inds])[::5])

#plt.savefig("sb-corrole-30vor-50ps-new.pdf")

plt.show()
# plt.show()
#%%
# K_c_graph_soft = networkx.from_numpy_matrix(K_c)
# networkx.draw(K_c_graph_soft, with_labels=True, font_weight='bold')
# plt.show()
#%%
check_commutator(K,nclus=5)
#%%
#memory 

Sc , detSc = rebinding(K, nclus=nclus)
_ , detSc_hard = rebinding(K, nclus=nclus, hard=True )
#%%
fast_picked = [ 0,  1,  2,  3,  4,  5,  7,  8, 10, 12, 16, 17, 19, 20, 22, 23, 24,
       25, 30, 32, 33, 39, 44, 47, 49, 50, 51, 52, 53, 54]
fast_chi = [0, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 1, 1, 1, 1, 1, 1, 1]
labels= ["A","B","C","D", "a","b", "c","d"]
plt.figure(figsize=(13,12))
cmap_states = plt.get_cmap("tab10")
plt.title("Sb-Corrole pump-probe specturm \n ex 400nm")#" \n Assignment of dominant conformaiton from PCCA+ \n %d Voronoi cells"%nstates)
plt.imshow(spectrum_infgen,cmap="coolwarm", aspect="auto")
plt.xlabel(r"$\lambda/$nm")
plt.ylabel("delay time [ps]")
plt.xticks(np.arange(len(wl), step=120),labels=np.round(wl[1::120]))

       #start with zero but remember to take it off from the lambdas in the data
plt.yticks(np.arange(0,1200, step=100))
#plt.yticks([0,10,20,30,40,50])       
plt.colorbar()
# for i in range(23,len(picked_inds)):
#     plt.axhline(y=picked_inds[i], color=cmap_states(np.argmax((chi_k)[i,:])), label=labels[np.argmax((chi_k)[i,:])])
# for i in range(30):
#     plt.axhline(y=fast_picked[i], color=cmap_states(fast_chi[i]+3), label=labels[fast_chi[i]+3], lw=3)
# plt.yscale("log")        
#plt.legend()
#plt.savefig("sb_corrole_all_thesis.pdf")
plt.show()
#%%
#transform K tens with pcca+
K_pcca = np.zeros((jumps,nclus,nclus))
for i in range(jumps):
    print(i)
    K_pcca[i] = proj(K_tens[i],nclus, pi="uniform")
infgen = infgen_3ways(K_pcca[:2] )
taus = []
for j in range(3):
    taus.append(1/infgen[j].diagonal())