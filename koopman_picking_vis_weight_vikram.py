#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:08:06 2020

@author: bzfsechi
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import svd, pinv, logm, eig
import cmdtools
from tools import plot_spectrum_strx, Koopman, stroboscopic_inds



#Here you load the data
#%%

# data_1 = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt').T
data_1 = np.loadtxt("br_py2_exec400.txt").T
#%%
#start 300 ps
spectrum_1 = data_1[1:, 45:]
ts1 = data_1[0,45:]
aaa = stroboscopic_inds(ts1)
wl = data_1[1:,0]
#%%

nclus = 5
jumps = 2
nstates = 20
#this is the function you need. it works with the picking algorithm
spectrum_infgen, picked_inds,centers, K_tens, indices, distances = Koopman(spectrum_1.T, ts1, w=10**7/wl, nstates=nstates, picked_weights=False)

#%%
K = K_tens[1] # transition matrix 
eig_k = np.sort(np.linalg.eigvals(K))
eigvec_k = np.linalg.eig(K)[1] # rapid check on the eigenvalues
print(eig_k)

plt.imshow(K)

plt.show()
plt.plot(eig_k, "-o")
plt.show()

#%%
chi_k = cmdtools.analysis.pcca.pcca(K,nclus, pi="uniform") # this is the pcca+ algorithm, matrix chi with teh membership functions

     #%%
     # with this you project into the clustered transition matrix K_c (here are two ways, but they are the same)
K_c = pinv(chi_k).dot(K.dot(chi_k))
K_c1 = pinv(chi_k.T.dot(chi_k)).dot(chi_k.T.dot(K.dot(chi_k)))


#%% 
#just plotting
color_list = ["r", "deepskyblue", "fuchsia", "gold","darkgreen","coral","black"]
plot_spectrum_strx(spectrum_1.T,wl, ts1)
for i in range(len(picked_inds)):
    plt.axhline(y=picked_inds[i], color=color_list[np.argmax((chi_k)[i,:])])
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
# #dass
DAS = pinv(chi_k).dot(centers)

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
