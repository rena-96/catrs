#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:36:57 2020

@author: bzfsechi
"""
import numpy as np
import matplotlib.pyplot as plt
from method_nmf import nmf
from scipy.linalg import logm 
from reduction_projection import rebinding_nmf
from infgen_4ways import infgen_3ways
#%%
#data = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt')
data = np.loadtxt("matrix_2.dat").T
#%%
spectrum = data[50:,1:]#before was 102 for the time
times = data[50:, 0]
wavelengths = data[0,1:]
#for i in range(spectrum.shape[0]):
#    spectrum[i,:]-=spectrum[95,:]
nclus = 3
# parameters = [0, -100., 100., 1., 10.]
parameters = [0., 100, 10, 1, 10]
#M_rec, W_rec, H_rec, P_rec, A_opt, Chi, UitGramSchmidt = nmf(spectrum,wavelengths, 4, 0, parameters, weight=True) 

M_rec, W_rec, H_rec, P_rec, A_opt, Chi, UitGramSchmidt = nmf(spectrum.T,r=nclus, params=parameters, weight=False) 


#%%
plt.figure(figsize=(15,4))
plt.suptitle('MF with PCCA+ analysis', fontsize=16)
labels= ["A","B","0"]
plt.subplot(1, 2, 1)
plt.title("Plot $H$")
for i in range(len(Chi.T)):
    plt.plot(H_rec[i], label=labels[i])
    #plt.xticks(np.arange(len(wavelengths), step=120),labels=(np.round(wavelengths[1::120]/1000)))
plt.grid()
plt.legend()
#plt.xlim(400,100) #flip the data
plt.xlabel("t[ps]")#"$\lambda$/nm")
plt.ylabel("concentration proportion")
plt.subplot(1, 2, 2)
plt.title("Plot $W$")
for i in range(len(Chi.T)):
    plt.plot(W_rec.T[i], label="compound_%s"%labels[i])
plt.xticks(np.arange(len(wavelengths), step=120),labels=(np.round(wavelengths[1::120]/1000)))
  #  plt.xticks(np.arange(len(data[0,1:]), step=50),labels=np.round(data[0,1::50],1))
#plt.xticks(np.arange(len(data[44:,0]), step=15),labels=np.round(data[44::15,0],1))
plt.legend()
plt.grid()
#plt.xlim(400,100)
plt.xlabel(r"$\nu/10^3$cm-1")
plt.ylabel("$\Delta A$")
#plt.savefig("bothh_w,process2_3clus.pdf")
plt.show()
#%%
#plt.figure(figsize=(14,16))
num = 321
for i in range(len(Chi.T)):
#    plt.figure(figsize=(10,4))
    
 #   plt.subplot(num)
    plt.plot(Chi.T[i], label="$\chi$_%d"%(i+1))
    
  #  plt.xticks(np.arange(len(data[0,1:]), step=30),labels=np.round(data[0,1::30]))
    plt.grid()
    plt.legend()
    plt.title("$\chi$")
#    plt.xlim(400,80) #flip the data
    plt.xlabel("$t$/ps")
    plt.ylabel("value of the column vector")
    num+=1
#plt.savefig("chi_3clust_process2.pdf")
plt.show()
    #%%
#infgen 
K_tensor = np.zeros((2,3,3))
K_tensor[0,:,:] = (H_rec.dot(H_rec.T))/ (H_rec.dot(H_rec.T)).sum(axis=1)[:, None]
#K_tensor[0,:,:] = np.eye(3)
K_tensor[1,:,:] = P_rec
#K_tensor[2,:,:] = 
infgen = infgen_3ways(K_tensor )
