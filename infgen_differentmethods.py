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
list_Koopman_gen = []
for types in center_type:
    K = voronoi.VoronoiTrajectory(spectrum, 20, centers=types).propagator() 
    list_Koopman_gen.append(K-np.eye(K.shape[0]))
list_ew = []
for i in range(3):
    list_ew.append(np.sort(np.linalg.eigvals(list_Koopman_gen[i])))
list_Chi = []
for c in range(3):
    list_Chi.append(cmdtools.analysis.pcca.pcca(list_Koopman_gen[c],5))
    #%%
for j in range(3):
    plt.imshow(list_Koopman_gen[j])
    plt.show()
    plt.plot(list_ew[j], "-o")
    plt.show()
    plt.imshow(list_Chi[j], aspect="auto")
    plt.show()
    #%%
Q_c = []
for i in range(3):
    Q_c.append( pinv(list_Chi[i]).dot(list_Koopman_gen[i].dot(list_Chi[i])))#/ (pinv(list_Chi[i]).dot(list_Chi[i])))
    print(np.sum(Q_c[i], axis =1))
#plt.imshow(Q_c)
    #%%
