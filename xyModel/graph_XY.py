#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhshang
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import sys

lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

input_set = sys.argv[1]

L=32
i=0
with open(input_set, 'r') as infile:
    for line in infile:
        if i % 2 == 0:
            T = line.split()[0]
            i+=1
        else:
            XY = lol(line.split(), L)
            XY = np.array(XY, dtype=float)
            # plot the network cluster
            [X, Y] = np.mgrid[0:L, 0:L]
            X_plot = np.cos(XY)
            Y_plot = np.sin(XY)

            plt.figure()
  #          plt.quiver(X, Y, X_plot, Y_plot, color='blue', pivot='mid')
            plt.quiver(X,Y,X_plot,Y_plot,XY,pivot='mid')
            plt.axis('equal')
            plt.axis('off')
          #  cbar = plt.colorbar(ticks=[-3.14, 0, 3.14])
           # cbar.ax.set_yticklabels(['-PI', '0', 'PI'])

            name = 'XY-T_'+str(T)
            figname = name+'.pdf'
            plt.savefig(figname, format='pdf', bbox_inches='tight')

            filename = name+'.data'
            np.savetxt(filename,XY)
            i+=1                               
