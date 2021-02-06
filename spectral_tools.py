# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 10:53:40 2021

@author: rena-96
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