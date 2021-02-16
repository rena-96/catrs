#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 13:16:57 2021

@author: bzfsechi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, pinv
from cmdtools.analysis import pcca
from tools import hard_chi
#%%
def statdistr(M):
    ews, levs = eig(M, left=True, right=False)
    inds = np.argsort(abs(ews.real))
    levs = levs[:,inds]
    stat_distr = levs[:,-1]
   #print(ews, levs)
    return(stat_distr.reshape(-1))
def proj(M, nclus, hard=False):
    pi = statdistr(M)
    dim = np.shape(M)[0]
    print("statdistr", pi)
    pi = np.ones((np.shape(M)[0]))
    chi = pcca.pcca(M, nclus)
    
    S_c = chi.T.dot(np.diag(pi).dot(chi))/(chi.T.dot(np.ones(dim)))

    T_c = chi.T.dot(np.diag(pi).dot(M.dot(chi)))/(chi.T.dot(np.ones(dim)))
    return(pinv(S_c).dot(T_c))
matrix = np.array([[0,1,0,0],[0.1,0.1,0.1,0.7],[0.,0.4,0.4,.2],[0.3,0.2,0.5,0.]])
distr = statdistr(matrix)
print(proj(matrix,3))