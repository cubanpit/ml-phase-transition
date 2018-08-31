import scipy
import matplotlib
matplotlib.use('Agg')
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import sys


lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

def saw(x):
    if x  <= - np.pi:
        return x + 2 * np.pi
    elif - np.pi <= x <= np.pi:
        return x
    elif np.pi <= x:
        return x - 2 * np.pi
def column(matrix, i):
        return [row[i] for row in matrix]

input_set = sys.argv[1]

L = 32    # Size
s = 2     # Submatrix size
i = 0
v = []    # Vorticity vector

graph = False   # Set True if you want to create vortex graphics

vortex_conf = open('vortex_L'+str(L)+'.dat', "a")

with open(input_set, 'r') as infile:
    for line in infile:
        if i % 2 == 0:
            T = line.split()[0]     # Store temperature
            vortex_conf.write(str(T)+"\n")
            i+=1
        else:
            v = []
            XY = lol(line.split(), L)      # Store configuration
            XY = np.array(XY, dtype=float)

            # Create L+1xL+1 matrix with PBC
            XY_P = np.column_stack((XY, column(XY, 0)))
            XY_P = np.vstack((XY_P, XY_P[0]))

            # Something to divide the LxL matrix in all the sxs submatrix
            P = L + 1 - s + 1
            x = np.arange(P).repeat(s)
            y = np.tile(np.arange(s), P) + x
            a = XY_P

            b = a[np.newaxis].repeat(P, axis=0)
            c = b[x, y]

            # Create submatrix
            d = c.reshape(P, s, L+1)
            e = d[:, np.newaxis].repeat(P, axis=1)
            f = e[:, x, :, y]
            g = f.reshape(P, s, P, s)
            h = g.transpose(2, 0, 3, 1)

#            print h

            # Calculate vorticity for each sxs submatrix
            for l in range(0, L):
                for j in range (0, L):
                    wn = np.round(saw(h[l][j].item((0,0))-h[l][j].item((0,1)))\
                            +saw(h[l][j].item((0,1))-h[l][j].item((1,1)))\
                            +saw(h[l][j].item((1,1))-h[l][j].item((1,0)))\
                            +saw(h[l][j].item((1,0))-h[l][j].item((0,0))),\
                            decimals=10)/np.round(2*np.pi, decimals=10)
                #    w_tmp = np.round(saw(h[l][j].item((0,0))-h[l][j].item((0,1)))\
                #            +saw(h[l][j].item((0,1))-h[l][j].item((1,1)))\
                #            +saw(h[l][j].item((1,1))-h[l][j].item((1,0)))\
                #            +saw(h[l][j].item((1,0))-h[l][j].item((0,0))),\
                #            decimals=10)

                #    if wn != 0.0:
                #        print("i= "+str(L-j)+" j= "+str(l+1))
                #        print wn
                #        print w_tmp
                #        print("saw(teta2-teta1)= "+str(saw(h[l][j].item((0,0))-h[l][j].item((0,1))))+ " teta1 (0,1)= "+str(h[l][j].item((0,1)))+" teta2(0,0)= "+str(h[l][j].item((0,0))))
                #        print("saw(teta3-teta2)= "+str(saw(h[l][j].item((1,0))-h[l][j].item((0,0))))+ " teta2 (0,0)= "+str(h[l][j].item((0,0)))+" teta3(1,0)= "+str(h[l][j].item((1,0))))
                #        print("saw(teta4-teta3)= "+str(saw(h[l][j].item((1,1))-h[l][j].item((1,0))))+ " teta3 (1,0)= "+str(h[l][j].item((1,0)))+" teta4(1,1)= "+str(h[l][j].item((1,1))))
                #        print("saw(teta1-teta4)= "+str(saw(h[l][j].item((0,1))-h[l][j].item((1,1))))+ " teta4 (1,1)= "+str(h[l][j].item((1,1)))+" teta1(0,1)= "+str(h[l][j].item((0,1))))
                #        print "\n"

                # Due to np.round, there are some "-0.0". Crappy workaround.
                    if str(wn) == "0.0" or str(wn) == "-0.0":
                       wn = 0.0

                    v.append(wn)        # Create vorticity array
            vortex_conf.write(str(" ".join( repr(e) for e in v))+'\n')

            if graph :
                # Convert array in LxL matrix (work only for square)
                V = lol(v, L)
                V = np.array(V, dtype=float)

                # Plot configuration and save it on .pdf
                [X, Y] = np.mgrid[0:L, 0:L]
                X_plot = np.cos(XY)
                Y_plot = np.sin(XY)

                plt.figure()
    #           ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
    #           plt.quiver(X, Y, X_plot, Y_plot, color='blue', pivot='mid')
                plt.quiver(X,Y,X_plot,Y_plot,V,pivot='mid')
                plt.axis('equal')
    #           plt.axis('off')
                cbar = plt.colorbar(ticks=[-1, 1])
    #           cbar.ax.set_yticklabels(['-PI', '0', 'PI'])

                name = 'XY-T_'+str(T)
                figname = name+'.pdf'
                plt.savefig(figname, format='pdf', bbox_inches='tight')

                filename = name+'.data'
                np.savetxt(filename,XY)
            i+=1
vortex_conf.close()

# Copyright 2018 Pietro F. Fontana <pietrofrancesco.fontana@studenti.unimi.it>
#                Martina Crippa    <martina.crippa2@studenti.unimi.it>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
