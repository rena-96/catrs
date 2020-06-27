#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:36:57 2020

@author: bzfsechi
"""
import numpy as np
import matplotlib.pyplot as plt
from method_nmf import nmf
#%%
data = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt')
#data = np.loadtxt("br_py2_exec400.txt")
trialmtx = data[1:,1:]
#for i in range(trialmtx.shape[0]):
#    trialmtx[i,:]-=trialmtx[95,:]

parameters = [-0.00001, -100., 100., .1, 10.]

M_rec, W_rec, H_rec, P_rec, A_opt, Chi, UitGramSchmidt = nmf(trialmtx[96:,:],4, parameters) 
#%%
plt.figure(figsize=(13,7))
plt.subplot(1, 2, 1)
plt.title("reconstructed spectrum with nmf")
plt.imshow(np.dot(W_rec,H_rec))
plt.xticks(np.arange(len(data[0,1:]), step=20),labels=np.round(data[0,1::20],1))
plt.yticks(np.arange(len(data[1:,0]), step=20),labels=np.round(data[1::20,0],1))
plt.subplot(1, 2, 2)
plt.title("real data")
plt.imshow(trialmtx)
plt.xticks(np.arange(len(data[0,1:]), step=20),labels=np.round(data[0,1::20],1))
plt.yticks(np.arange(len(data[1:,0]), step=20),labels=np.round(data[1::20,0],1))
#plt.colorbar()
plt.show()

#%%
plt.figure(figsize=(10,5))
plt.imshow(abs(np.dot(W_rec,H_rec)-trialmtx[96:,:]))
plt.colorbar()
plt.show()
print('max error:', np.amax(abs(np.dot(W_rec,H_rec)-trialmtx[96:,:])), 'min error:', np.amin(abs(np.dot(W_rec,H_rec)-trialmtx[96:,:])))
plt.imshow(H_rec, aspect= "auto", interpolation="nearest")
plt.show()
plt.imshow(W_rec, aspect= "auto", interpolation="nearest")

#%%
plt.imshow(Chi.T, aspect= "auto", interpolation= "nearest")
plt.imshow(H_rec, aspect= "auto", interpolation="nearest")

#%%
plt.figure(figsize=(14,4))
plt.subplot(1, 2, 1)
plt.title("Plot $\chi$")
for i in range(len(Chi.T)):
    plt.plot(Chi.T[i], label=i)
    plt.xticks(np.arange(len(data[0,1:]), step=20),labels=np.round(data[0,1::20],1))
#    plt.show()
plt.legend()
plt.xlim(100,0) #flip the data
plt.xlabel("$\lambda$/nm")
plt.ylabel("value of the column vector")
plt.subplot(1, 2, 2)
plt.title("Plot $H_{rec}$")
for i in range(len(Chi.T)):
    plt.plot(H_rec[i], label=i)
    plt.xticks(np.arange(len(data[0,1:]), step=20),labels=np.round(data[0,1::20],1))
plt.legend()
plt.xlim(100,0)
plt.xlabel("$\lambda$/nm")
plt.ylabel("value of the column vector")
plt.show()