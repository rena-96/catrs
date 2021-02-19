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
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import cmdtools
from cmdtools import utils
from cmdtools.estimation.picking_algorithm import picking_algorithm
#%%

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
    # return(np.searchsorted(x, np.arange(np.max(x)+1),side="right")-1)
    return(np.searchsorted(x, np.arange(np.max(x)),side="right"))    

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


    
    
def plot_spectrum_strx(X, ls,ts, step_=60, strobox=True):
    """X: spectrum, ls=wavelengths,ts=time"""
    if strobox==True:
        strobox = stroboscopic_inds(ts)
        
        X_new = X[strobox,:]
        ts_new = ts[strobox]
        step_ = int(len(ts_new)/10)
        plt.figure(figsize=(7,6))
        plt.imshow(X_new, cmap="coolwarm",aspect = "auto", alpha=0.8)
        plt.colorbar()
        plt.title("Pump-probe spectrum of brominated \n aluminium corrole exec.400nm")
        plt.xlabel("$\lambda$ [nm]")
        plt.ylabel("delay time [ps]")
        plt.xticks(np.arange(len(ls), step=step_),labels=np.round(ls[1::step_]))
        #start with zero but remember to take it off from the lambdas in the data
        plt.yticks(np.arange(len(ts_new), step=step_),labels=np.round(ts_new[::step_],2))
        
    
    
        
    elif strobox==False:
        X_new = X
        ts_new = ts
        step_ = int(len(ts)/10)
        plt.figure(figsize=(7,6))
        plt.imshow(X_new, cmap="coolwarm",aspect = "auto", alpha=0.8)
        plt.colorbar()
        plt.title("Pump-probe spectrum of brominated \n aluminium corrole exec.400nm")
        plt.xlabel("$\lambda$ [nm]")
        plt.ylabel("delay time [ps]")
        plt.xticks(np.arange(len(ls), step=60),labels=np.round(ls[1::60]))
        #start with zero but remember to take it off from the lambdas in the data
        plt.yticks(np.arange(len(ts_new), step=step_),labels=np.round(ts_new[::step_],2))
      
        
   # plt.show()
    
        
    
    
    

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

def hard_chi(X_vecs):
    """Tranform the chi vectors with indicator function
    ---write better descr"""
    X_new = np.zeros(X_vecs.shape)
    maxs = np.argmax((X_vecs), axis=1)
    for i in range(X_vecs.shape[0]):
        X_new[i,maxs[i]] = 1.
    return(X_new)
        
def Koopman(X,timeseries,w,nstates=50,jumps=10, picked_weights=False):
    strobox = stroboscopic_inds(timeseries)
    
    X_strbx = (X)[strobox,:]
    K_tens = np.zeros((jumps,nstates, nstates))
    w_sqrt = np.sqrt(w)
    if picked_weights==False:
        picked_inds = np.sort(picking_algorithm(X_strbx,nstates)[1])
    else:
        picked_inds = np.sort(picking_algorithm(w_sqrt*X_strbx,nstates)[1])
    centers = X_strbx[picked_inds,:]
    # if w==None:
        # print("Hello")
  
    dist = distance.cdist(X_strbx, centers,metric="sqeuclidean")
    inds1 =  np.argmin(dist, axis=1)
    # else:
    
    w_dist = distance.cdist(w_sqrt*X_strbx, w_sqrt*centers,metric="sqeuclidean")
    inds =  np.argmin(w_dist, axis=1)
    # # print(inds, "inds of K")
   # print(w_dist==dist)
    for j in range(jumps):
        
        for i in range(0,len(inds)-j):
            (K_tens[j])[inds[i], inds[i+j]] += 1
            
        K_tens[j] = utils.rowstochastic(K_tens[j])
    return(X_strbx, picked_inds,centers, K_tens, [inds, inds1],[w_dist,dist])
def nn_weighted(A,B,W):
    inds =  (NearestNeighbors(metric="wminkowski", metric_params={"w":W})
          .fit(B).kneighbors(A, 1, False)
          .reshape(-1))
    return(inds)