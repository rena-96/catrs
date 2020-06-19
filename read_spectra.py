#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:15:12 2020

@author: bzfsechi
"""

import numpy as np
import matplotlib.pyplot as plt
#%%
spectrum1 = np.loadtxt('br_py2_exec400')
spectrum2 = np.loadtxt('iso_br_al_cor_py2_400nm_ex_ir')
#%%
xtick = spectrum1[0,1:]
plt.imshow(spectrum1[1:,1:], aspect='auto')
plt.show()
plt.imshow(spectrum2[1:,1:], aspect='auto')
plt.colorbar()
plt.show()
#plt.xticks(spectrum1[0,1:])
#plt.yticks(spectrum1[1:,0])