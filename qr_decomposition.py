# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:43:12 2020

@author: renat
"""

import numpy as np
import matplotlib.pyplot as plt
#%%
#follows ideas in https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0017373#s1
# QR decomposition to find the spectra as exponentials
spectrum = np.loadtxt("br_py2_exec400.txt")[1:,1:]
q,r = np.linalg.qr(spectrum)
#%%
#for i in  range(4):
#    plt.plot(q[:,i])