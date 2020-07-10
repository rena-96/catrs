#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:04:04 2020

@author: bzfsechi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, pinv
from scipy.optimize import fmin
import cmdtools
from tools import norm_rows, avg_spectrum
#%%


def three_states_system(M):
    M_copy = np.zeros(M.shape)
    th = 0.8
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i,j]>th:
                M_copy[i,j] = 1.
                
            elif abs(M[i,j])<th:
                M_copy[i,j] = 2.
            else:
                M_copy[i , j] = 3.
    return(M_copy)
def countmatrix(M, avg=1, tau=1):
    M = norm_rows(M, avg)
    tau = int(tau)
    count = M[:-tau,:].T.dot(M[tau:,:])
    return norm_rows(count)
#
#a = np.random.rand(4,4) - np.random.rand(4,4)
#count_a = countmatrix(a)
#%%
#trialmtx = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt')[1:,1:]
data = np.loadtxt("br_py2_exec400.txt")
trialmtx = data[1:,1:]
red_mtx = three_states_system(0.15*trialmtx[:,:])
#trialmtxpos = trialmtx-np.amin(trialmtx)
plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.imshow(trialmtx[:,:], cmap='inferno', aspect='auto')
plt.colorbar()
plt.title("spectrum")
plt.xticks(np.arange(len(data[0,1:]), step=50),labels=np.round(data[0,1::50],1))
plt.yticks(np.arange(len(data[1:,0]), step=20),labels=np.round(data[1::20,0],1))
plt.subplot(1, 2, 2)
plt.imshow(red_mtx, cmap="inferno_r",aspect = "auto")
plt.colorbar()
plt.title("reduced three states system")
plt.xticks(np.arange(len(data[0,1:]), step=50),labels=np.round(data[0,1::50],1))
plt.yticks(np.arange(len(data[1:,0]), step=20),labels=np.round(data[1::20,0],1))
plt.show()
count_tm = countmatrix(red_mtx[101:,:], 1,1)
plt.imshow(count_tm)
plt.title("count matrix from 3 states system, $\tau=1$")
plt.colorbar()
plt.show()
#%%
#mtx2 = trialmtx.T.dot(trialmtx)
#plt.imshow(countmatrix(mtx2, avg=3))

#plt.show()
#%%
Chi = cmdtools.pcca(count_tm, 4)
#plt.imshow(chi_k, cmap="inferno", aspect= "auto", interpolation= "nearest")
#plt.title("$\chi$ from count matrix")
#plt.colorbar()
#plt.show()
for i in range(len(Chi.T)):
    plt.plot(Chi.T[i], label=i)
    plt.xticks(np.arange(len(data[0,1:]), step=50),labels=np.round(data[0,1::50],1), label=i)
    plt.legend()
    plt.xlim(400,100)
    plt.xlabel("$\lambda/nm$")
    plt.title("$\chi$")
    plt.show()
#%%
Pc = np.linalg.pinv(Chi).dot(count_tm.dot(Chi))
