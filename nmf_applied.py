#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:36:57 2020

@author: bzfsechi
"""
import numpy as np
import matplotlib.pyplot as plt
from method_nmf import nmf
#%%

trialmtx = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt')[1:,1:]

parameters = [-0.00001, -100., 100., .1, 10.]

M_rec, W_rec, H_rec, P_rec, A_opt, Chi, UitGramSchmidt = nmf(trialmtx,4, parameters) 
plt.subplot(1, 2, 1)
plt.imshow(np.dot(W_rec,H_rec))
plt.subplot(1, 2, 2)
plt.imshow(trialmtx)
plt.colorbar()
plt.show()
#%%
plt.imshow(abs(np.dot(W_rec,H_rec)-trialmtx))
plt.colorbar()
print('max error:', np.amax(abs(np.dot(W_rec,H_rec)-trialmtx)), 'min error:', np.amin(abs(np.dot(W_rec,H_rec)-trialmtx)))
plt.imshow(H_rec, aspect= "auto", interpolation="nearest")
plt.show()
plt.imshow(W_rec, aspect= "auto", interpolation="nearest")
#%%
#for i in range(4):
#    plt.plot(chi_k.T.dot(W_rec[i,:]))
#    #%%
#dd = pinv(chi_k).dot(trialmtx.T)
#plt.imshow(dd,aspect="auto")
#plt.show()
#plt.imshow(H_rec, aspect= "auto")
#%%