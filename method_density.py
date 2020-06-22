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
#%%
def avg_spectrum(M, avg):
    check_divisibility =  M.shape[1]%avg
    if not int(check_divisibility)==0:
        raise AssertionError("Try another value for the number of lambdas to average upon")

    avg_matrix = np.zeros((M.shape[0],int(M.shape[1]/avg)))
    for i in range(int(M.shape[1]/avg)):
        avg_matrix[:,i] = np.mean(M[:,i*avg:(i+1)*avg], axis=1)
    return(avg_matrix)
def norm_rows(M, avg=1):
    '''Norm easily first to make every wavelength at everytime 
    of equal importance'''
    M = avg_spectrum(M,avg)
    return M/np.sum(M, axis =0)

def three_states_system(M):
    M_copy = np.zeros(M.shape)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i,j]>0.2:
                M_copy[i,j] = 1.
                
            elif abs(M[i,j])<0.01:
                M_copy[i,j] = 2.
            else:
                M_copy[i , j] = 3.
    return(M_copy)
def countmatrix(M, avg=1):
    M = norm_rows(M, avg)
    count = M[:-1,:].T.dot(M[1:,:])
    return norm_rows(count)
#
#a = np.random.rand(4,4) - np.random.rand(4,4)
#count_a = countmatrix(a)
#%%
trialmtx = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt')[1:,1:]
red_mtx = three_states_system(trialmtx)
#trialmtxpos = trialmtx-np.amin(trialmtx)
plt.imshow(trialmtx, cmap='inferno', aspect='auto')
plt.colorbar()
plt.title("spectrum")
plt.show()
plt.imshow(red_mtx, cmap="tab10", aspect = "auto")
plt.colorbar()
plt.title("reduced three states system")
plt.show()
count_tm = countmatrix(red_mtx, 1)
plt.imshow(count_tm)
plt.title("count matrix from 3 states system, $\tau=1$")
plt.colorbar()
plt.show()
#%%
#mtx2 = trialmtx.T.dot(trialmtx)
#plt.imshow(countmatrix(mtx2, avg=3))

#plt.show()
#%%
chi_k = cmdtools.pcca(count_tm, 4)
plt.imshow(chi_k, cmap="inferno", aspect="auto")
plt.title("$\chi$ from count matrix")
plt.colorbar()
plt.show()
