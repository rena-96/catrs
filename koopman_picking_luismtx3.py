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
from tools import  plot_spectrum_strx, stroboscopic_inds
import cmdtools.estimation.voronoi as voronoi
from cmdtools import utils
from cmdtools.estimation.picking_algorithm import picking_algorithm
from cmdtools.estimation.newton_generator import Newton_N
from sklearn.neighbors import NearestNeighbors
from infgen_4ways import infgen_3ways
from reduction_projection import proj, rebinding
#%%

data_1 = np.loadtxt("matrix_2.dat")
# data_2 = np.loadtxt("matrix_2.dat")
# data_3 = np.loadtxt("matrix_3.dat")
#data_1 = np.loadtxt('iso_br_al_cor_py2_420nm_ex_ir')
#%%
spectrum_1 = data_1[1:, 50:]
ts1 = data_1[0,50:]
aaa = stroboscopic_inds(ts1)

#%%
# K, spectrum_new, picked_inds = voronoi_koopman_picking(spectrum_1.T,20,timeseries=data_1[0,102:],dt=1)
#%%
#infgen
nclus = 3
jumps = 3
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
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.show()
plt.plot(eig_k, "-o")
plt.show()
plt.imshow(chi_k, aspect="auto")
plt.show()
     #%%
K_c = pinv(chi_k).dot(K.dot(chi_k))
K_c1 = pinv(chi_k.T.dot(chi_k)).dot(chi_k.T.dot(K.dot(chi_k)))
K_c2 = proj(K,nclus, pi="statdistr")
S_c, detSc = rebinding(K, nclus=nclus)
T_c = S_c.dot(K_c) 
#     print(np.sum(K_c[i], axis =1))
plt.imshow(K_c)
plt.colorbar()
#     #%%
color_list = ["g",  "deepskyblue", "fuchsia", "gold","ivory","darkgreen","coral"]
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
    plt.plot(aaa[picked_inds],chi_k[:,i], "-o", label="$\chi$_%d"%(i+1)) 
    plt.grid()
    plt.legend()
   # plt.title("$\chi$ vectors")
    plt.ylabel("$\chi$ value (membership)")
    plt.xlabel("time/ps")
#    plt.xticks(ticks=np.arange(len(aaa), step=1000),labels=aaa[::1000])
#plt.savefig("process2_chi2_50.pdf")
plt.show()
#%%
plt.imshow(spectrum_infgen, cmap="coolwarm", aspect="auto")
plt.xlabel(r"$\nu/10^3$cm-1")
plt.ylabel("delay time [ps]")
plt.xticks(np.arange(len(data_1[1:,0]), step=150),labels=np.round(data_1[1::150,0]/1000))
        #start with zero but remember to take it off from the lambdas in the data
plt.yticks([1000,3000,5000,7000,9000])
plt.title("Simulated spectrum")   
plt.colorbar()
#plt.savefig("process2_spectrum.eps")
plt.show()
#%%
# for i in [0,1,2]:
#     #plt.plot(ts1,Chi[:,i], label="$NMF-\chi$_%d"%i)
#     plt.plot(aaa[picked_inds],chi_k[:,i],"-o",label="$MSM-\chi$_%d"%(i+1))
#     plt.xlabel("delaytime/ps")
#     plt.ylabel("\chi")
# plt.grid()
# plt.legend()
# #plt.savefig(args, kwargs)
DAS = pinv(chi_k).dot(centers)
plt.figure(figsize=(18,6))
plt.suptitle("Membership functions $\chi$ and species \n-product ansatz")
plt.subplot(1,2,1)


labels = ["0","B","A","C","D","E", "F","G"]
for i in range(chi_k.shape[1]):
    plt.plot(ts1[aaa[picked_inds]],chi_k[:,i], "-o", label=labels[i])#"$\chi$_%d"%i) 
    
    plt.legend()
    plt.title(r"$\chi$ of $K(\tau)$")
    plt.ylabel("$\chi$ value (membership)")
    plt.xlabel("time/ps")
    
plt.grid()  
plt.xscale("linear")

plt.subplot(1,2,2)  
#plt.xticks(ticks=aaa[::15])#, labels=(aaa[picked_inds])[::5])
for i in range(chi_k.shape[1]):
    #plt.plot(ts1,Chi[:,i], label="$NMF-\chi$_%d"%i)
    plt.plot(data_1[95:,0]*0.001,DAS[i,94:],"-.",label="$MSM-S$_%s"%labels[i])
    plt.xlabel(r" $\nu/10^3$cm-1")
    plt.title("Compounds Amplitudes")
plt.grid()
plt.legend()
#plt.savefig("seq3clustmsm.pdf")

plt.show()
#%%
# for i in [50,280]:
#     plt.plot(wl,spectrum_infgen[i,:], "-", label="$\chi$_%d"%(i+1)) 
#     plt.grid()
#     plt.legend()
#    # plt.title("$\chi$ vectors")
#     plt.ylabel("$\chi$ value (membership)")
#     plt.xlabel("time/ps")
# #    plt.xticks(ticks=np.arange(len(aaa), step=1000),labels=aaa[::1000])
# #plt.savefig("process2_chi2_50.pdf")
# plt.show()
#transform K tens with pcca+
K_pcca = np.zeros((jumps,nclus,nclus))
for i in range(jumps):
    
    K_pcca[i] = proj(K_tens[i],nclus, pi="uniform")
infgen = infgen_3ways(K_pcca)
#%%
lalala =  cmdtools.analysis.pcca.pcca(K-np.eye(50),nclus)
lelele =  pinv(lalala).dot((K-np.eye(50)).dot(lalala))