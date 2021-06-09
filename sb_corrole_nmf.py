#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:36:57 2020

@author: bzfsechi
"""
import numpy as np
import matplotlib.pyplot as plt
from method_nmf import nmf
from scipy.linalg import logm, svd
from reduction_projection import rebinding_nmf
from tools import stroboscopic_inds
#%%
# data_1 = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt').T
file = np.load("SB_corrole_400nm.npz")
data_1 = file["data"][0,:,:,0,:]
wl = file["wl"][:,0]
ts = file["t"]

#%%
spectrum = np.nanmean(data_1, axis=2)
#start analysis at 300 fs and 300 ps  and  400-700 nm circa
ts = ts[46:(183+46)]*0.001
wl = wl[500:1448]
spectrum = spectrum[46:(183+46),500:1448]
ts_inds = stroboscopic_inds(ts)
spectrum = spectrum[ts_inds]
nclus = 5
# parameters = [0, -100., 100., 1., 10.]
parameters = [0., 100, 10, 10, 1.]


M_rec, W_rec, H_rec, P_rec, A_opt, Chi, UitGramSchmidt = nmf(spectrum.T,r=nclus, params=parameters, weight=False) 


#%%
plt.figure(figsize=(15,4))
plt.suptitle('MF with PCCA+ analysis', fontsize=16)
labels= ["$c_1$","$c_2$","$c_3$", "$c_4$", "$c_5$", "F", "G"]
color_list = ["r", "deepskyblue", "fuchsia", "gold","darkgreen","coral","black"]
cmap = plt.get_cmap("tab10")
plt.subplot(1, 2, 1)
plt.title("Plot $H$")
for i in range(len(Chi.T)):
    plt.plot(ts[ts_inds],H_rec[i], '--', color=cmap((i)),label=labels[i])
    #plt.xticks(np.arange(len(wl), step=120))#,labels=(np.round(wl[/1000)))
plt.grid()
plt.legend()
#plt.xlim(400,100) #flip the data
plt.xlabel("t[ps]")#"$\lambda$/nm")
plt.ylabel("concentration proportion")
plt.subplot(1, 2, 2)
plt.title("Plot $W$")
for i in [1,2,3,4,0]:
    
    plt.scatter(np.arange(len(wl)),W_rec.T[i], marker=".", color=cmap(i), label=labels[i])
plt.xticks(np.arange(len(wl), step=100),labels=(np.round(wl[1::100])))
plt.xlim(948,0)
  #  plt.xticks(np.arange(len(data[0,1:]), step=50),labels=np.round(data[0,1::50],1))
#plt.xticks(np.arange(len(data[44:,0]), step=15),labels=np.round(data[44::15,0],1))
plt.legend()
plt.grid()
#plt.xlim(400,100)
plt.xlabel(r"$\lambda$[nm]")
plt.ylabel("$\Delta A$")
#plt.savefig("sb-corrole-nmf-5clus-35ps.pdf")
plt.show()
#%%
#plt.figure(figsize=(14,16))
num = 321
for i in range(len(Chi.T)):
#    plt.figure(figsize=(10,4))
    
 #   plt.subplot(num)
    plt.plot(Chi.T[i], label="$\chi$_%d"%(i+1))
    
  #  plt.xticks(np.arange(len(data[0,1:]), step=30),labels=np.round(data[0,1::30]))
    plt.grid()
    plt.legend()
    plt.title("$\chi$")
#    plt.xlim(400,80) #flip the data
    plt.xlabel("$t$/ps")
    plt.ylabel("value of the column vector")
    num+=1
#plt.savefig("chi_3clust_process2.pdf")
plt.show()
    #%%
S_mf, detSmf = rebinding_nmf(H_rec.T)

#%%
#from scipy.optimize import curve_fit
##plt.plot(((-data[41:-1,0]+data[42:,0])),"-o")
#
#fit = curve_fit(lambda t,a,b,c: a+b*np.exp(c*t),  W_rec[50:,2],  data[94:,0])

diagp = np.diagonal(-1/logm(P_rec))
for i in range(len(Chi.T)):
    fig, ax = plt.subplots()
    ax.plot(W_rec.T[i])
    plt.xticks(np.arange(len(wl), step=100),labels=(np.round(wl[1::100])))
    ax.invert_xaxis()
    plt.ylim(np.amin(W_rec),np.amax(W_rec))
    plt.grid()
    plt.show()