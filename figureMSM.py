#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 14:21:01 2021

@author: bzfsechi
"""

import numpy as np
import cmdtools
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
#%%
dims = 2
step_n = 1000
step_set = [-0.1, 0., 0.1]
origin = np.zeros((1,dims))# Simulate steps in 2D
step_shape = (step_n,dims)
steps = np.random.choice(a=step_set, size=step_shape)
path = np.concatenate([origin, steps]).cumsum(0)
start = path[:1]
stop = path[-1:]

#%%
#picked centers
picked, _,d = cmdtools.estimation.picking_algorithm.picking_algorithm(path, 5)
inds = np.argmin(d, axis=1)

#%%
#voronoi ccells
vor = Voronoi(picked)
#%%
# Plot the path
#fig = plt.figure(figsize=(20,10))

fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='black', line_width=2, line_alpha=0.6, point_size=2)
ax = fig.add_subplot(111)

#ax.scatter(path[:,0], path[:,1],c="b",alpha=0.45,s=1 );
ax.plot(path[:,0], path[:,1],c="black", alpha=0.8,ls="-.")#,lw=0.25,ls="-");
ax.scatter(picked[:,0], picked[:,1], c="red")
# ax.plot(start[:,0], start[:,1],c="r", marker="s")
ax.plot(stop[:,0], stop[:,1],c="black", marker="s")
plt.xlim(np.amin(path[:,0])-0.1,np.max(path[:,0])+0.1)
plt.ylim(np.amin(path[:,1])-0.1,np.max(path[:,1])+0.1)
plt.title("2D-trajectory with Voronoi discretization")
ax.set_facecolor('#fff6d5')
plt.xticks([])
plt.yticks([])
plt.xlabel(" $\lambda_1$")
plt.ylabel("$\lambda_2$")
# plt.tight_layout(pad=0)
plt.savefig("random_walk_2d.png")
plt.show()
#%%
fig2 = plt.figure(figsize=(8,5))
ax2 = fig2.add_subplot(111)
ax2.scatter(np.arange(len(inds)), inds, c="black",marker=".")
# plt.xticks([])
plt.yticks(ticks=np.arange(5), labels=["A","B","C","D","E"])
plt.xlabel(" simulation frame")
plt.ylabel("assigned Voronoi cell ")
ax2.set_facecolor('#fff6d5')
plt.savefig("random_walk_2d_voronoijumps.png")
plt.show()
#%%
