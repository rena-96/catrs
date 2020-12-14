# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 18:16:47 2020

@author: renat
"""
import numpy as np
import matplotlib.pyplot as plt
n=np.linspace(10000,40000,1000) 

gESA_1a=np.exp(-(n-30000)**2/(2*2500)**2) 
gESA_1b=np.exp(-(n-20000) **2/(2*2500)**2) 
gbleach_1=-np.exp(-(n-25000) **2/(2*2500)**2) 
gSE_1=-np.exp(-(n-23000) **2/(2*3000)**2) 

gESA_2a=np.exp(-(n-32000) **2/(2*1000)**2) 
gESA_2b=np.exp(-(n-20000) **2/(2*5000)**2) 
gbleach_2=-np.exp(-(n-25000) **2/(2*2500)**2) 
gSE_2=-np.exp(-(n-12000) **2/(2*3000)**2) 

gESA_3a=np.exp(-(n-35000) **2/(2*5000)**2) 
gESA_3b=np.exp(-(n-25000) **2/(2*5000)**2) 
gbleach_3=-np.exp(-(n-30000) **2/(2*2500)**2) 
gSE_3=-np.exp(-(n-20000) **2/(2*3000)**2) 

gESA_4a=np.exp(-(n-35000) **2/(2*3000)**2) 
gESA_4b=np.exp(-(n-20000) **2/(2*3000)**2) 
gbleach_4=-np.exp(-(n-25000) **2/(2*2500)**2) 
gSE_4=-np.exp(-(n-12000) **2/(2*3000)**2) 

S1=3*gESA_1a+gbleach_1+gSE_1+0.4*gESA_1b 
S2=3*gESA_2a+gbleach_2+0.3*gESA_2b #+gSE_2, Triplet-artig Spektrum, ohne SE
S3=2*gESA_3a+gbleach_3+0.1*gESA_3b+gSE_3  # Ein anderes Bleach Spektrum
S4=3*gESA_4a+gbleach_4+0.3*gESA_4b+gSE_4 
#%%
plt.plot(n/10**3,S1, label="S1")
plt.plot(n/10**3,S2, label="S2")
plt.plot(n/10**3,S3, label="S3")
plt.plot(n/10**3,S4, label="S4")
plt.grid()
