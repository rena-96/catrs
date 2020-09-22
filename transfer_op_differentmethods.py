#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:08:06 2020

@author: bzfsechi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, pinv
import cmdtools
#from tools import norm_rows, avg_spectrum
import cmdtools.estimation.voronoi as voronoi
from cmdtools import utils
from cmdtools.estimation.picking_algorithm import picking_algorithm
#%%
#trialmtx = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt')[1:,1:]
data = np.loadtxt("br_py2_exec400.txt")
#data = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt')
spectrum = data[40:, 1:]
#%%
picked_ind = np.sort(picking_algorithm(spectrum, 30)[1])
center_type = ["kmeans", spectrum[picked_ind,:], spectrum[::5,:]]
list_Koopman = []
for types in center_type:
    K = voronoi.VoronoiTrajectory(spectrum, 20, centers=types).propagator() 
    list_Koopman.append(K)
list_ew = []
for i in range(3):
    list_ew.append(np.sort(np.linalg.eigvals(list_Koopman[i])))
list_Chi = []
for c in range(3):
    list_Chi.append(cmdtools.analysis.pcca.pcca(list_Koopman[c],5))
    #%%
for j in range(3):
    plt.imshow(list_Koopman[j])
    plt.show()
    plt.plot(list_ew[j], "-o")
    plt.show()
    plt.imshow(list_Chi[j], aspect="auto")
    plt.show()
    #%%
K_c = []
for i in range(3):
    K_c.append( pinv(list_Chi[i]).dot(list_Koopman[i].dot(list_Chi[i])))#/ (pinv(list_Chi[i]).dot(list_Chi[i])))
    print(np.sum(K_c[i], axis =1))
#plt.imshow(K_c)
    #%%
