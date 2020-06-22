#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:12:50 2020

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
def three_states_system(M, avg):
    """ neg = 1, zero = 2, pos = 3"""
    M_avg = np.round(avg_spectrum(M, avg),3)
    M_new = np.zeros(M_avg.shape)
    for i in range
#%%
trialmtx = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt')[1:,1:]

smt = avg_spectrum(trialmtx, 4)
#%%
plt.subplot(1, 2, 1)
plt.imshow(smt, aspect='auto')
plt.subplot(1, 2, 2)
plt.imshow(trialmtx, aspect='auto')
plt.colorbar()
plt.show()
