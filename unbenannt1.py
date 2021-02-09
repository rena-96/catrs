# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 07:22:33 2021

@author: renat
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import svd, pinv, logm
from scipy.spatial import distance
from sklearn.cluster import KMeans
import cmdtools
from tools import plot_spectrum_strx, Koopman, stroboscopic_inds, hard_chi, nn_weighted
import cmdtools.estimation.voronoi as voronoi
from cmdtools import utils
from cmdtools.estimation.picking_algorithm import picking_algorithm
from cmdtools.estimation.newton_generator import Newton_N
from sklearn.neighbors import NearestNeighbors
import networkx
from check_commutator import check_commutator
#%%

# data_1 = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt').T
data_1 = np.loadtxt("br_py2_exec400.txt").T
#%%
spectrum_1 = data_1[1:, 50:]
ts1 = data_1[0,50:]
aaa = stroboscopic_inds(ts1)
wl = data_1[1:,0]
#%%
#infgen
nclus = 5
jumps = 3
nstates = 50

# spectrum_infgen, picked_inds,centers, K_tens1 = Koopman(spectrum_1, ts1,jumps=jumps, w=1/wl)
# K_tens = np.zeros((jumps,nstates,nstates))
# for j in range(jumps):
    
        
#     for i in range(0,len(inds)-j):
#         (K_tens[j])[inds[i], inds[i+j]] += 1
    
# K_tens[j] = utils.rowstochastic(K_tens[j])

# def weighted_metric(A,B,W, p=2):
#     """
#     Use a Minkowski distance with p=2, i.e. Euclidean distance. 
#     A = timeseries,
#     B= matrix with centers
#     W = weights (frequencies)"""
    
#     dist = cdist(A, B, metric="minkowski",p=2, w=W)
#     return(dist)


# def nn_weighted(A,B,W):
#     inds =  (NearestNeighbors(metric="wminkowski", metric_params={"w":W})
#           .fit(B).kneighbors(A, 1, False)
#           .reshape(-1))
#     return(inds)
    
a = np.array([[0,0,0],[1,1,1],[2,2,2]])
b = np.array([[1,1,1],[2,2,2]])
W = np.sqrt(np.array([1,2,3]))
pc = distance.cdist(W*a,W*b,metric="sqeuclidean")
print(np.argmin(pc,axis=1))
