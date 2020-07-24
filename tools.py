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
import cmdtools
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

#def norm_vis(M, lambdas):
#    max_lambda = np.amax(lambdas[:-1]-lambdas[1:])
#    count = 0
#    while count < len(lambdas):
#        diff = 0
#    for i in range(last):
#        while abs(diff- max_lambda) > 0.3:
#            
#        
#    return new_M

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
