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
import cmdtools.estimation.voronoi as voronoi
from cmdtools import utils
from cmdtools import p
import cmdtools.estimation.picking_algorithm as picking_algorithm
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
    
    return(ind_matrix)
#def sum_trafo(M, centers):
#    sum_to = np.zeros()
#    for i in range(M.shape[1]):
        
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
#data = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt')
trialmtx = data[40:,1:]
red_mtx = three_states_system(0.15*trialmtx[:,:])
#trialmtxpos = trialmtx-np.amin(trialmtx)
plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.imshow(trialmtx[:,:], cmap='inferno', aspect='auto')
plt.colorbar()
plt.title("spectrum")
plt.xticks(np.arange(len(data[0,1:]), step=50),labels=np.round(data[0,1::50],1))
plt.yticks(np.arange(len(data[40:,0]), step=20),labels=np.round(data[40::20,0],1))
plt.subplot(1, 2, 2)
plt.imshow(red_mtx, cmap="inferno_r",aspect = "auto")
plt.colorbar()
plt.title("reduced three states system")
plt.xticks(np.arange(len(data[0,1:]), step=50),labels=np.round(data[0,1::50],1))
plt.yticks(np.arange(len(data[42:,0]), step=20),labels=np.round(data[40::20,0],1))
plt.show()


#%%
thing = picking_algorithm.picking_algorithm(red_mtx,10 )
trafo_pa = voronoi.VoronoiTrajectory(red_mtx, red_mtx.shape[1], centers=red_mtx[::10,:]).propagator()
#%%

plt.imshow(trafo_pa)
plt.title("transfer matrix from 3 states system, $\tau=1$")
plt.colorbar()
plt.show()
#%%

#temp = np.zeros(red_mtx.shape)
#for i in thing[1]:
#    temp[i,:] = red_mtx[i,:]
#    #%%
#plt.imshow(red_mtx+2*temp, cmap="tab10",aspect = "auto")
#plt.colorbar()
##lt.imshow(temp, cmap="gray",aspect = "auto")
    

transferop = voronoi.VoronoiTrajectory(trialmtx, trialmtx.shape[1], centers=trialmtx[::10,:]).propagator()   
#%%
ew, ev = np.linalg.eig(transferop)
#%%
plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.imshow(transferop, cmap='inferno_r', aspect='auto')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(trafo_pa, cmap="inferno_r",aspect = "auto")
plt.colorbar()
plt.show()
#%%
a = cmdtools.analysis.pcca.scipyschur(trafo_pa,5)
plt.imshow(a, aspect="auto")