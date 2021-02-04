# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:58:16 2021

@author: renat
"""
import numpy as np
from scipy.linalg import pinv
import cmdtools
def check_commutator(X, nclus=3):
    """function to check if the projection and the propagation commute
    G(Xc**k)=G(Xc)**k"""
    #projection
    chi = cmdtools.analysis.pcca.pcca(X,nclus)
    X_c = pinv(chi).dot(X.dot(chi))
    #propagation
    condition = 0
    power = 2
    while condition==0 and power<10:
        X_pw = np.linalg.matrix_power(X, power)
        chi_pw = cmdtools.analysis.pcca.pcca(X_pw,nclus)
        X_pw_c = pinv(chi_pw).dot(X_pw.dot(chi_pw)) 
        if np.linalg.matrix_power(X_c,power).all()==X_pw_c.all():
            condition = 1
        else: 
            power = power+1
    return(X_c, X_pw_c, power)
    

    
    