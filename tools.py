#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:32:33 2020

@author: bzfsechi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, pinv
from scipy.optimize import fmin
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import cmdtools
from cmdtools import utils
from cmdtools.estimation.picking_algorithm import picking_algorithm
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


def norm_IR( lambdas):
    new_M = []
    max_lambda = np.amax(lambdas[:-1]-lambdas[1:])
    for i in range(len(lambdas)-1):
        diff = abs(lambdas[i]-lambdas[i+1])
        if abs(diff - max_lambda)<= 0.2*max_lambda:
            new_M.append(i)
        else:
            continue
    return(new_M)
    
def pi_pcca(lambdas):
    """Compute the weights of the time-resolved spectrum for a specific 
    value of lambda to consider in the PCCA+ algorithm. The smallest 
    wavelength difference is taken as unity and the differences between 
    walengths are scaled to that smallest value. Return the density pi. """
    diff = abs(lambdas[:-1]-lambdas[1:])
    min_lambda = np.amin(diff)
    pi = np.zeros((len(diff)+1))
    pi[0] = diff[0]/min_lambda
    pi[-1] = diff[-1]/min_lambda
    for i in range(1, len(pi)-1):
        pi[i] = (diff[i-1]+diff[i])/(2*min_lambda)
    return pi/np.sum(pi)
#%%
def weight_spectrum(M, lambdas):
    """Repeat same column multiple times so that is has the same weight"""
    dens = pi_pcca(lambdas)
    dens = np.around(dens/np.amin(dens))
    weighted_M = np.zeros((M.shape[0], int(np.sum(dens))))
    count = 0  
    for i in range(len(dens)):
        copies = dens[i]
        for j in range(int(copies)):
            weighted_M[:, int(count+j)] = M[:,i]
        count+=copies
    return(weighted_M)
def weight_time(dt_array):
    time_intervals = np.log(abs(dt_array[:-1]-dt_array[1:]))
    step_first = 1-time_intervals[0]
    return(time_intervals+step_first)

def stroboscopic_inds(x):
    return(np.searchsorted(x, np.arange(np.max(x)+1),side="right")-1)
    

def voronoi_propagator(X, centers, nstates, dt):
    P = np.zeros((nstates, nstates))
    if centers == "kmeans":
        k = KMeans(n_clusters=nstates).fit(X)
        inds = k.labels_
   # centers = k.cluster_centers_
    else:
        
        inds =  (NearestNeighbors()
            .fit(centers).kneighbors(X, 1, False)
            .reshape(-1))
    if len(dt)==1:
        for i in range(len(inds)-dt[0]):
            P[inds[i], inds[i+dt[0]]] += 1
    else:
        for i in range(len(inds)-1):
            time_weight = weight_time(dt)
            P[inds[i], inds[i+1]] += 1*time_weight[i]
        
    return utils.rowstochastic(P)
def voronoi_koopman(X, centers,nstates, timeseries, dt):
    strobox = stroboscopic_inds(timeseries)
    X_new = X[strobox,:]
    K = np.zeros((nstates, nstates))
    if centers == "kmeans":
        k = KMeans(n_clusters=nstates).fit(X_new)
        inds = k.labels_
   # centers = k.cluster_centers_
    else:
        
        inds =  (NearestNeighbors()
            .fit(centers).kneighbors(X_new, 1, False)
            .reshape(-1))
    select_inds = stroboscopic_inds(timeseries) #tau=1
   
    for i in range(0,len(inds)-1, dt):
            K[inds[i], inds[i+1]] += 1
    print(K)
    return utils.rowstochastic(K)

def voronoi_koopman_picking(X, nstates, timeseries, dt):
    strobox = stroboscopic_inds(timeseries)
    X_new = X[strobox,:]
    K = np.zeros((nstates, nstates))
    centers = X_new[np.sort(picking_algorithm(X_new,20)[1]),:]
    inds =  (NearestNeighbors()
            .fit(centers).kneighbors(X_new, 1, False)
            .reshape(-1))
    #tau=1
   
    for i in range(0,len(inds)-1, int(dt)):
            K[inds[i], inds[i+1]] += 1
    #print(K)
    return utils.rowstochastic(K)
    
    
    
    
        
    
    
    

def analyse_spectrum_picking_alg(spectrum, timesteps, no_centers):
    picked_ind = np.sort(picking_algorithm(spectrum, int(no_centers))[1])
    Koopman = voronoi_propagator(spectrum,spectrum[picked_ind, :], int(no_centers),dt = timesteps)
    eigvals = np.sort(np.linalg.eigvals(Koopman))
    plt.plot(eigvals, "-o")
    plt.show()
    n_c = input("How many dominants eigenvalues?")
    Chi_ = cmdtools.analysis.pcca.pcca(Koopman,int(n_c))
    K_c = pinv(Chi_).dot(Koopman.dot(Chi_))
    return(K_c, Chi_, picked_ind)

