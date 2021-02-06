# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 07:22:33 2021

@author: renat
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

def weighted_metric(A,B,W, p=2):
    """
    Use a Minkowski distance with p=2, i.e. Euclidean distance. 
    A = timeseries,
    B= matrix with centers
    W = weights (frequencies)"""
    
    dist = cdist(A, B, metric="minkowski",p=2, w=W)
    return(dist)


def nn_weighted(A,B,W):
    inds =  (NearestNeighbors(metric="wminkowski", metric_params={"w":W})
          .fit(B).kneighbors(A, 1, False)
          .reshape(-1))
    return(inds)
    
# a = np.array([[0,0,0],[0,0,0],[2,2,2]])
# b = np.array([[1,1,1],[2,2,2]])
# w = np.array([1,2,3])
# pc = weighted_metric(a,b,w)
print(weighted_metric(a,b,w),nn_weighted(a, b, w) )
