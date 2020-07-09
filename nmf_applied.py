#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:36:57 2020

@author: bzfsechi
"""
import numpy as np
import matplotlib.pyplot as plt
from method_nmf import nmf
from tools import norm_rows
#%%
#data = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt')
data = np.loadtxt("br_py2_exec400.txt")
trialmtx = data[1:,1:]
#for i in range(trialmtx.shape[0]):
#    trialmtx[i,:]-=trialmtx[95,:]

parameters = [0, -100., 100., 1., 10.]

M_rec, W_rec, H_rec, P_rec, A_opt, Chi, UitGramSchmidt = nmf(trialmtx[43:,:],5, parameters) 
#%%
#plt.figure(figsize=(13,7))
#plt.subplot(1, 2, 1)
#plt.title("reconstructed spectrum with nmf")
#plt.imshow(np.dot(W_rec,H_rec))
#plt.xticks(np.arange(len(data[0,1:]), step=50),labels=np.round(data[0,1::50],1))
#plt.yticks(np.arange(len(data[1:,0]), step=20),labels=np.round(data[1::20,0],1))
#plt.subplot(1, 2, 2)
#plt.title("real data")
#plt.imshow(trialmtx)
#plt.xticks(np.arange(len(data[0,1:]), step=50),labels=np.round(data[0,1::50],1))
#plt.yticks(np.arange(len(data[1:,0]), step=20),labels=np.round(data[1::20,0],1))
##plt.colorbar()
#plt.show()

#%%
plt.figure(figsize=(10,5))
#plt.imshow(abs(np.dot(W_rec,H_rec)-trialmtx[43:,:])/trialmtx[43:,:])
#plt.colorbar()
#plt.show()
print('max error:', np.amax(abs(np.dot(W_rec,H_rec)-trialmtx[43:,:])), 'min error:', np.amin(abs(np.dot(W_rec,H_rec)-trialmtx[43:,:])))
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
plt.grid()
plt.legend()
plt.xlim(400,100) #flip the data
plt.xlabel("$\lambda$/nm")
plt.ylabel("value of the column vector")
plt.subplot(1, 2, 2)
plt.title("Plot $H_{rec}$")
for i in range(len(Chi.T)):
    plt.plot(H_rec[i], label=i)
    plt.xticks(np.arange(len(data[0,1:]), step=50),labels=np.round(data[0,1::50],1))
plt.legend()
plt.xlim(400,100)
plt.xlabel("$\lambda$/nm")
plt.ylabel("value of the column vector")
plt.show()
#%%
for i in range(len(Chi.T)):
    plt.figure(figsize=(10,4))
    plt.plot(Chi.T[i], label=i)
    plt.xticks(np.arange(len(data[0,1:]), step=20),labels=np.round(data[0,1::20]))
    plt.grid()
    plt.legend()
    plt.xlim(400,80) #flip the data
    plt.xlabel("$\lambda$/nm")
    plt.ylabel("value of the column vector")
    plt.show()
    #%%
plt.plot(trialmtx[179,:]/np.sum(trialmtx[179,:])*60)
plt.plot(Chi.T[3,:]-0.5)
plt.xlim(400,80)
#%%
plt.plot(trialmtx[118,:]/np.sum(trialmtx[118,:])*60)
plt.plot(Chi.T[0,:]-0.3)
plt.xlim(400,80)
#%%
#for j in [44,120,160,179]:
#    plt.plot(trialmtx[j,:]/np.sum(trialmtx[j,:])*60)
#%%
plt.figure(figsize=(10,4))
for i in [0,1,2,3,4]:
   
    plt.plot(Chi.T[i], label=i)
    plt.xticks(np.arange(len(data[0,1:]), step=20),labels=np.round(data[0,1::20]))

plt.legend()
plt.xlim(400,80) #flip the data
plt.xlabel("$\lambda$/nm")
plt.ylabel("value of the column vector")
plt.grid()
plt.show()
#%%
for i in [0,1,2,3,4]:
   # plt.figure(figsize=(10,4))
    plt.plot(W_rec.T[i], label=i)
plt.legend()
from scipy.linalg import svd
U,S,V = svd(trialmtx)
#%%
#from scipy.optimize import curve_fit
#plt.plot(((-data[41:-1,0]+data[42:,0])),"-o")

#fit = curve_fit(lambda t,a,b: a+b*np.exp(t),  np.arange(141),  data[40:,0])

