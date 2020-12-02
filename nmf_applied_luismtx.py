#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:36:57 2020

@author: bzfsechi
"""
import numpy as np
import matplotlib.pyplot as plt
from method_nmf import nmf
from scipy.linalg import logm 
from tools import norm_rows
#%%
#data = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt')
data = np.loadtxt("matrix_3.dat").T
#%%
spectrum = data[100:,1:]#before was 102 for the time
times = data[100:, 0]
wavelengths = data[0,1:]
#for i in range(spectrum.shape[0]):
#    spectrum[i,:]-=spectrum[95,:]

# parameters = [0, -100., 100., 1., 10.]
parameters = [0., 100, 10, 1, 10]
#M_rec, W_rec, H_rec, P_rec, A_opt, Chi, UitGramSchmidt = nmf(spectrum,wavelengths, 4, 0, parameters, weight=True) 

M_rec, W_rec, H_rec, P_rec, A_opt, Chi, UitGramSchmidt = nmf(spectrum.T,r=2, params=parameters, weight=False) 
#%%
#plt.figure(figsize=(13,7))
#plt.subplot(1, 2, 1)
#plt.title("reconstructed spectrum with nmf")
#plt.imshow(np.dot(W_rec,H_rec))
#plt.xticks(np.arange(len(data[0,1:]), step=50),labels=np.round(data[0,1::50],1))
#plt.yticks(np.arange(len(data[1:,0]), step=20),labels=np.round(data[1::20,0],1))
#plt.subplot(1, 2, 2)
#plt.title("real data")
#plt.imshow(spectrum)
#plt.xticks(np.arange(len(data[0,1:]), step=50),labels=np.round(data[0,1::50],1))
#plt.yticks(np.arange(len(data[1:,0]), step=20),labels=np.round(data[1::20,0],1))
##plt.colorbar()
#plt.show()

#%%
# plt.figure(figsize=(10,5))
# #plt.imshow(abs(np.dot(W_rec,H_rec)-spectrum)/spectrum)
# #plt.colorbar()
# #plt.show()
# #print('max error:', np.amax(abs(np.dot(W_rec,H_rec)-spectrum)), 'min error:', np.amin(abs(np.dot(W_rec,H_rec)-spectrum)))
# plt.imshow(H_rec, aspect= "auto", interpolation="nearest")
# plt.show()
# plt.imshow(W_rec, aspect= "auto", interpolation="nearest")

#%%
# plt.imshow(Chi.T, aspect= "auto", interpolation= "nearest")
# plt.imshow(H_rec, aspect= "auto", interpolation="nearest")

#%%
plt.figure(figsize=(15,4))
plt.suptitle('NMF&PCCA+ analysis from 0ps, mech3', fontsize=16)
plt.subplot(1, 2, 1)
plt.title("Plot $\chi$")
for i in range(len(Chi.T)):
    plt.plot(Chi.T[i], label=i)
    #plt.xticks(np.arange(len(wavelengths), step=120),labels=(np.round(wavelengths[1::120]/1000)))
plt.grid()
plt.legend()
#plt.xlim(400,100) #flip the data
plt.xlabel("t[ps]")#"$\lambda$/nm")
plt.ylabel("$\chi$[i]")
plt.subplot(1, 2, 2)
plt.title("Plot $W_{rec}$")
for i in range(len(Chi.T)):
    plt.plot(W_rec.T[i], label=i)
plt.xticks(np.arange(len(wavelengths), step=120),labels=(np.round(wavelengths[1::120]/1000)))
  #  plt.xticks(np.arange(len(data[0,1:]), step=50),labels=np.round(data[0,1::50],1))
#plt.xticks(np.arange(len(data[44:,0]), step=15),labels=np.round(data[44::15,0],1))
plt.legend()
plt.grid()
#plt.xlim(400,100)
plt.xlabel(r"$\nu/10^3$cm-1")
plt.ylabel("$W_{rec}$[i]")
plt.show()
#%%
plt.figure(figsize=(14,16))
num = 321
for i in range(len(Chi.T)):
#    plt.figure(figsize=(10,4))
    
    plt.subplot(num)
    plt.plot(Chi.T[i], label=i)
  #  plt.xticks(np.arange(len(data[0,1:]), step=30),labels=np.round(data[0,1::30]))
    plt.grid()
    plt.legend()
    plt.title("$\chi$_%d"%i)
#    plt.xlim(400,80) #flip the data
    plt.xlabel("$\lambda$/nm")
    plt.ylabel("value of the column vector")
    num+=1
plt.show()
    #%%
#plt.figure(figsize=(10,4))
#for i in [0,1,2,3,4]:
#   
#    plt.plot(H_rec[i], label=i)
#    #plt.xticks(np.arange(len(data[0,1:]), step=20),labels=np.round(data[0,1::20]))
#plt.title("$\chi$,analysis from 250 fs")
##plt.legend()
##plt.xlim(400,80) #flip the data
#plt.xlabel("$\lambda$/nm")
#plt.ylabel("value of the column vector")
#plt.xticks(np.arange(len(data[44:,0]), step=15),labels=np.round(data[44::15,0],1))
#plt.legend()
#plt.grid()
##plt.xlim(400,100)
#plt.xlabel("t/ps")
#
#plt.show()
#%%
# for i in range(len(Chi.T)):
#     plt.plot(abs(H_rec[i]))
#%%
#from scipy.optimize import curve_fit
##plt.plot(((-data[41:-1,0]+data[42:,0])),"-o")
#
#fit = curve_fit(lambda t,a,b,c: a+b*np.exp(c*t),  W_rec[50:,2],  data[94:,0])

diagp = np.diagonal(-1/logm(P_rec))
