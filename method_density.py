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


def countmatrix(M, avg=1):
    M = norm_rows(M, avg)
    count = M[:-1,:].T.dot(M[1:,:])
    return norm_rows(count)
a = np.random.rand(4,4) - np.random.rand(4,4)
count_a = countmatrix(a)
#%%
trialmtx = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt')[1:,1:]

count_tm = countmatrix(trialmtx, avg=2)
plt.imshow(count_tm)
plt.colorbar()
plt.show()
#%%
mtx2 = trialmtx.T.dot(trialmtx)
plt.imshow(countmatrix(mtx2, avg=3))
plt.show()