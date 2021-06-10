#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:07:46 2021

@author: bzfsechi
"""



import numpy as np
import matplotlib.pyplot as plt
from method_nmf import nmf
from scipy.linalg import svd
import csv


spectrum=np.loadtxt('spec_matrix.csv',delimiter=',')
print(spectrum.shape)
energy = spectrum[:,0]
spectrum = spectrum[:,1:]
#%%
#spectrum = spectrum[20000:, 10:]
#%%

plt.imshow(spectrum, aspect="auto")
plt.colorbar()
#plt.yticks(np.arange(len(energy), step=1000), labels=energy[::1000])
plt.show()
#%%
U, S, V = svd(spectrum, full_matrices=False)
#%%
parameters = [-1000, 100., 200., 1.,1.]
M_r, W_r, H_r, P_r, A_optimized, chi, Uitgm = nmf(spectrum, r=3, params=parameters)

print(np.amin(H_r), np.amin(W_r))

#%%
for i in range(4):
    plt.plot( H_r[i,:])

     #%%
plt.figure(figsize=(12,12))
for i in range(4):
    
    plt.plot(energy, W_r[:,i])