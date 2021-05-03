# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:32:41 2021

@author: R.Sechi
This code is to compute the infinitesimal generator
"""
import numpy as np
from scipy.linalg import logm
from scipy.optimize import fmin
from cmdtools.estimation.newton_generator import Newton_N
import matplotlib.pyplot as plt

def logm_infgen(K_tensor, tau=1):
    tau = float(tau)
    return(logm(K_tensor[tau,:,:])/tau)

def fin_diff_infgen(K_tensor, tau=1):
    tau = float(tau)
    return (K_tensor[tau,:,:]-K_tensor[0,:,:])/tau


def to_minimize(Q,H):
    #Q = fin_diff_infgen(K_tensor)
    diff = 0
    for tau in range(H.shape[1]):
         diff += np.linalg.norm(H[:,tau]- np.exp(Q*tau)*H[0,:])
    return(diff)
def nlls_infgen(H,K_tensor, tau=1):
    """vectors= chi vectors for MSM and H vectors for NMF"""
    Q0=fin_diff_infgen(K_tensor, tau)
    Q_flattered = fmin(to_minimize, x0=Q0, args=(H))
    dim =  int(np.sqrt(Q_flattered.shape[0]))
    return Q_flattered.reshape(dim,dim)
    
    
# def 4ways_infgen(K_tensor, tau=1):
#     """Compute infgen in 4 ways. 
#     K_tensor= tensor, it starts w K for tau=0"""
#     return [logm_infgen(K_tensor, tau), fin_diff_infgen(K_tensor, tau), nlls_infgen(K_tensor, tau), Newton_N(K_tensor, 1, 0)]
    