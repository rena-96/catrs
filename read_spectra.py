#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:15:12 2020

@author: bzfsechi
"""

import numpy as np
import matplotlib.pyplot as plt
#%%
spectrum1 = np.loadtxt('br_py2_exec400.txt')
spectrum2 = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir.txt')
#%%
xtick = spectrum1[0,1:]
plt.imshow(spectrum1[1:,1:], aspect='auto')
plt.show()
plt.imshow(spectrum2[1:,1:], aspect='auto')
plt.colorbar()
plt.show()
#plt.xticks(spectrum1[0,1:])
#plt.yticks(spectrum1[1:,0])
#%%
diff_l = abs(spectrum2[0,1:-1]-spectrum2[0,2:])
def norm_IR( lambdas):
    new_M = []
    max_lambda = np.amax(abs(lambdas[:-1]-lambdas[1:]))
    count = 0
#    for i in range(count,len(lambdas)-1):
    while count < (len(lambdas)-1):
        
        diff = abs(lambdas[count]-lambdas[count+1])
       # print(abs(diff))
        if abs(diff - max_lambda)<= 0.4*max_lambda:
            new_M.append(count)
            count+=1
        else:
            new_M.append(count+1)
            count+=2
    return(new_M)

    
new_ir = norm_IR(spectrum2[0,1:])
new_diff = abs(spectrum2[0,new_ir[:-1]]-spectrum2[0,new_ir[1:]])