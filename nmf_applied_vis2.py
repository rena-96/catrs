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
from tools import  plot_spectrum_strx
#%%
data= np.loadtxt("br_py2_exec400.txt")
#%%
spectrum = data[50:,1:]#before was 102 for the time
times = data[50:, 0]
wavelengths = data[0,1:]
#for i in range(spectrum.shape[0]):
#    spectrum[i,:]-=spectrum[95,:]
nclus = 5
# parameters = [0, -100., 100., 1., 10.]
parameters = [0., 100, 10, 1, 10]
#M_rec, W_rec, H_rec, P_rec, A_opt, Chi, UitGramSchmidt = nmf(spectrum,wavelengths, 4, 0, parameters, weight=True) 

M_rec, W_rec, H_rec, P_rec, A_opt, Chi, UitGramSchmidt = nmf(spectrum.T,r=nclus, params=parameters, weight=False) 


#%%
labels = ["A","B","C","D","E"]
plt.figure(figsize=(15,4))

plt.suptitle('NMF&PCCA+ analysis vis spectrum', fontsize=16)
plt.subplot(1, 2, 1)
plt.title("Plot $H_{rec}$")
for i in range(len(Chi.T)):
    plt.plot(times,H_rec[i], label=labels[i])
 #   plt.xticks(times[::10],labels=np.round(times[::10],2))
plt.grid()
plt.legend()
#plt.xscale("symlog")
#plt.xlim(400,100) #flip the data
plt.xlabel("t[ps]")#"$\lambda$/nm")
plt.ylabel("concentration")
plt.subplot(1, 2, 2)
plt.title("Plot $W_{rec}$")
for i in range(len(Chi.T)):
    plt.plot(wavelengths,W_rec.T[i], label=labels[i])
    
# plt.xticks(wavelengths),labels=(np.round(wavelengths)[::60]))
  #  plt.xticks(np.arange(len(data[0,1:]), step=50),labels=np.round(data[0,1::50],1))
#plt.xticks(np.arange(len(data[44:,0]), step=15),labels=np.round(data[44::15,0],1))
plt.legend()
plt.grid()
plt.xlim(440,850)
plt.xlabel(r"$\lambda$nm")
plt.ylabel("$\Delta A$")
plt.show()
#%%
plt.figure(figsize=(14,16))
num = 321
for i in range(len(Chi.T)):
#    plt.figure(figsize=(10,4))
    
    plt.subplot(num)
    plt.plot(times,Chi.T[i], label=i)
    
  #  plt.xticks(np.arange(len(data[0,1:]), step=30),labels=np.round(data[0,1::30]))
    plt.grid()
    plt.legend()
    plt.title("$\chi$_%d"%i)
#    plt.xlim(400,80) #flip the data
    plt.xlabel("$t$/ps")
    plt.ylabel("value of the column vector")
    num+=1
plt.show()
#     #%%
# plot_spectrum_strx(spectrum, wavelengths, times, strobox=False)
# color_list = ["g", "ivory", "deepskyblue", "fuchsia", "gold","darkgreen","coral"]
# reg_lines = spectrum[::10,:]
# for i in range(len(times[::10])-1):
#     plt.axhline(y=(times[::10])[i])# color=color_list[np.argmax((H_rec[::10,:])[i])])
# # plt.savefig("pcca_nt_br_al_corr_vis.pdf")
# plt.show()
#%%
# for i in range(len(Chi.T)):
#     plt.plot(abs(H_rec[i]))
#%%
#from scipy.optimize import curve_fit
##plt.plot(((-data[41:-1,0]+data[42:,0])),"-o")
#
#fit = curve_fit(lambda t,a,b,c: a+b*np.exp(c*t),  W_rec[50:,2],  data[94:,0])

diagp = np.diagonal(-1/logm(P_rec))
#%%
maxx = []
for i in range(131):
    maxx.append(np.argmax(H_rec[:,i]))
    print(maxx[i],i)
    
    #%%

plt.grid()
plt.scatter(times, maxx, marker=".")
plt.ylabel("state")
plt.xlabel("t[ps]")
plt.title("Dominant state with MF w PCCA+ \nfrom 500fs and 5 states")
# plt.xscale("log")
# plt.xticks(ticks= times[::10])

