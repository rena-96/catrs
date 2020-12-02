#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:22:07 2020

@author: bzfsechi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, diagsvd,pinv
from scipy.optimize import fmin
import cmdtools
from tools import pi_pcca



def make_Uit(M, r, weight):
    '''Obtain the matrix denoted as U italics with the leading 
    r-1 vectors and the constant vector e=[1,1,...1] as first column.
    Input: 
        M=arr, matrix with the spectrum
        r=int, number of species
        weight= kwarg, if False take the left singular values, if True 
        weightes them with the singular values
    Output:
        s= ordered singular values
        U= left singular vectors
        Uit= constant vec + r-1 dominant left singular vectors '''
    U, s, _ = svd(M.T)
    if weight==False:
        Uit = np.vstack((np.ones(np.shape(U)[0]), U[:, :(r-1)].T))
    else:
        Utilde = (U[:,:r]).dot(diagsvd(s[:r],r, M.T.shape[0]))
        Uit = np.vstack((np.ones(np.shape(U)[0]), Utilde[:, :(r-1)].T))
    return(s, U, Uit.T)
    
def PSI_2(A, Uit, M, params):
    """Return the penalty funxction PSI^2
    Input:
    A: rotation matrix from pcca+
    Uit: matrix with dominant vectors
    M: spectrum
    params: parameters for the penalties
    Output:
        float: square of the penalty function PSI
        """
    Uit_p = Uit[1:, :]
    Uit_m = Uit[:-1, :]
    shape = int(np.sqrt(A.shape[0]))
    A = A.reshape(shape, shape)
    P = np.dot(pinv(np.dot(Uit_m, A)), np.dot(Uit_p, A))
    pen1 = params[0]*np.amin(np.dot(M, pinv(np.dot(Uit, A).T)))
    pen2 = params[1]*np.amin(np.dot(Uit, A).T)
    pen3 = params[2]*np.amax(abs(np.sum(np.dot(Uit, A).T, axis=0)-1))
    pen4 = params[3]*np.amin(P)
    pen5 = params[4]*np.amax(abs(np.sum(P, axis=0)-1))
    return (pen1+ pen2 + pen3 + pen4 + pen5)**2

def find_Aopt(A, Uit, M, params):
    '''Minimize PSI^2 and return the optimized rotation matrix A'''
    A_opt_flattered = fmin(PSI_2, x0=A, args=(Uit, M, params))
    dim =  int(np.sqrt(A_opt_flattered.shape[0]))
    return A_opt_flattered.reshape(dim,dim)

def pcca_Umodified(Uit, lambdas, dens=1):
    """pcca+ with the modified dominant left sing vectors"""
    m = np.shape(Uit)[0]
    if dens==1:
        pi = np.ones(m)*1/float(m)
    else:
        pi = pi_pcca(lambdas)
    #print("dasist pi", pi )
    Uit = cmdtools.analysis.pcca.gramschmidt(Uit, pi)
    optim = cmdtools.analysis.optimization.Optimizer()
    A = optim.solve(Uit, pi)
    chi = np.dot(Uit, A)
    return chi, Uit, A
   
 

def nmf(M, lambdas=1, r = 3, dens=1, params = [1,1,1,1,1], weight=False):
    """Method of NMF withouth separability assumption. Use notation&method 
    described in 'Analyzing Raman Spectral Data without Separability
    Assumption', Konstantin Fackeldey, Jonas Röhm, Amir Niknejad, 
    Surahit Chewle, Marcus Weber, 2020"""
    _, _, Uit = make_Uit(M, int(r), weight)
    chi, Uitgm, A = pcca_Umodified(Uit, lambdas, dens)
    A_optimized = find_Aopt(A, Uitgm, M, params)
    H_r = np.dot(Uitgm, A_optimized).T
    H_r_mt = np.dot(Uitgm[:-1,:], A_optimized)
    H_r_pt = np.dot(Uitgm[1:,:], A_optimized)
    W_r = np.dot(M, pinv(H_r))
    P_r = np.dot(pinv(H_r_mt),H_r_pt)
    M_r = np.dot(W_r, H_r)
    return(M_r, W_r, H_r, P_r, A_optimized, chi, Uitgm )    