#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:09:35 2020

@author: Alexander Sikorski
"""
import numpy as np
import matplotlib.pyplot as plt

Q1 = np.array([[-1,1,0],[0,-100,100],[10,0,-10]])
Q2 = np.array([[-1,1,0],[1, -11, 10], [0,10,-10]])

def gillespie(Q, x0, n_iter):
    x = x0
    xs = [x]
    ts = [0]
    for i in range(n_iter):
        rate = np.sum(Q[x,:]) - Q[x,x]
        tau = np.random.exponential(1/rate)
        q = Q[x,:] / rate
        q[x] = 0
        x = np.random.choice(range(len(q)), p=q)

        ts.append(ts[-1]+tau)
        xs.append(x)

    return np.array(xs), np.array(ts)

xs, ts = gillespie(Q2, 0, 100)

plt.scatter(ts, xs)