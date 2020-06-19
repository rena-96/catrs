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

def norm_rows(M):
    '''Norm easily first to make every wavelength at everytime 
    of equal importance'''
    return M/np.sum(M, axis =0)


def countmatrix(M):
    M = norm_rows(M)
    count = M[:-1,:].T.dot(M[1:,:])
    return norm_rows(count)
a = np.random.rand(4,4) - np.random.rand(4,4)
count_a = countmatrix(a)
#%%
trialmtx = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir')[1:,1:]

count_tm = countmatrix(trialmtx)
plt.imshow(count_tm)
plt.colorbar()