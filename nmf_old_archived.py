#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:22:07 2020

@author: bzfsechi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, pinv
from scipy.optimize import fmin
import cmdtools
#%%


def make_Uit(M, r):
    '''Obtain the matrix denoted as U italics with the leading 
    r-1 vectors and the constant vector e=[1,1,...1] as first column.
    Input: 
        M=arr, matrix with the spectrum
        r=int, number of species
    Output:
        s= ordered singular values
        U= left singular vectors
        Uit= constant vec + r-1 dominant left singular vectors '''
    U, s, _ = svd(M.T)
    Uit = np.vstack((np.ones(np.shape(U)[0]), U[:, :(r-1)].T))
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

def pcca_Umodified(Uit):
    """pcca+ with the modified dominant left sing vectors"""
    m = np.shape(Uit)[0]
    pi=np.ones(m)*1/float(m)
    Uit = cmdtools.analysis.pcca.gramschmidt(Uit, pi)
    optim = cmdtools.analysis.optimization.Optimizer()
    A = optim.solve(Uit, pi)
    chi = np.dot(Uit, A)
    return chi, Uit, A
   
 
#%%
data = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt')
trialmtx = data[1:,1:]

ss, Us, Uits = make_Uit(trialmtx,6)
#%%
parameters = [-0.00001, -100., 100., .1, 10.]


chis, Uitsgm, As = pcca_Umodified(Uits)
#%%
Asopt = find_Aopt(As, Uitsgm, trialmtx, parameters)

#%%
H_rec = np.dot(Uitsgm, Asopt).T
W_rec = np.dot(trialmtx,pinv(H_rec))
P_rec = np.dot(pinv(np.dot(Uitsgm[:-1,:], Asopt)), np.dot(Uitsgm[1:,:], Asopt))
#%%
plt.subplot(1, 2, 1)
plt.imshow(np.dot(W_rec,H_rec))
plt.subplot(1, 2, 2)
plt.imshow(trialmtx)
plt.colorbar()
plt.show()
#%%
plt.imshow(abs(np.dot(W_rec,H_rec)-trialmtx))
plt.colorbar()
print('max error:', np.amax(abs(np.dot(W_rec,H_rec)-trialmtx)), 'min error:', np.amin(abs(np.dot(W_rec,H_rec)-trialmtx)))
plt.imshow(H_rec, aspect= "auto")
plt.show()
plt.imshow(W_rec, aspect= "auto")
#%%
#for i in range(4):
#    plt.plot(chi_k.T.dot(W_rec[i,:]))
#    #%%
#dd = pinv(chi_k).dot(trialmtx.T)
#plt.imshow(dd,aspect="auto")
#plt.show()


    