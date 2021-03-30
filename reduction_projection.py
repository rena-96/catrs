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
def get_pi(M, pi="uniform"):
    if pi == "uniform":
        dim = np.size(M, 1)
        pi = np.full(dim, 1./dim)
    elif pi == "statdistr":
        ews, levs = eig(M, left=True, right=False)
        inds = np.argsort(abs(ews.real))
        levs = levs[:,inds]
        pi = levs[:,-1]
 
    return pi/pi.sum()


def proj(M, nclus, pi="uniform"):
  
    dim = np.shape(M)[0]
   
   # pi = np.ones((dim))*1/dim
    pi = get_pi(M, pi=pi)
    chi = pcca.pcca(M, nclus)
    
    S_c = chi.T.dot(np.diag(pi).dot(chi))#/(chi.T.dot(np.diag(pi).dot(np.ones(dim))))

    T_c = chi.T.dot(np.diag(pi).dot(M.dot(chi)))#/(chi.T.dot(np.diag(pi).dot(np.ones(dim))))
    return(pinv(S_c).dot(T_c))
# matrix = np.array([[0,1,0,0],[0.1,0.1,0.1,0.7],[0.,0.4,0.4,.2],[0.3,0.2,0.5,0.]])
# distr = statdistr(matrix)
# print(proj(matrix,3))
def rebinding(M, nclus, pi="uniform"):
    """Rebinding, paper marcus and max 12-13"""
    dim = np.shape(M)[0]
    pi = get_pi(M, pi=pi)
    #print(pi)
    chi = pcca.pcca(M, nclus)
    num = chi.T.dot(np.diag(pi).dot(chi))
    den = chi.T.dot(np.diag(pi).dot(np.ones((dim,1))))
    den = den*np.eye(nclus)
   # print(num.shape, den.shape)
    S_c = pinv(den).dot(num)
    return(S_c, np.linalg.det(S_c))
    
def rebinding_nmf(H_r):#, pi="uniform"):
    """h_r=chi, so the stiffness matrix is given by 
    h_r.T.pi.h_r/(h_r.T.pi.e),
    TODO: stiffness with initial distribution"""
    # if pi =="uniform":
       
    #     pi = np.full(dim, 1./dim)  
    # elif pi=="statdistr":
      
    rank = H_r.shape[1] 
    dim = H_r.shape[0]
    pi = np.full(dim, 1./dim)   
    print(rank, dim)
    #num = H_r.T.dot(np.diag(pi).dot(H_r))
   # den = H_r.T.dot(np.diag(pi).dot(np.ones((dim,1))))
    S_c = H_r.T.dot(H_r)
    den = den*np.eye(rank)
    print(num.shape, den.shape)
    S_c = pinv(den).dot(num)
    return(S_c, np.linalg.det(S_c), )
        
    