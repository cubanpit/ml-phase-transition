#! /bin/python3

# File: vortex.py

# This script should identify vortices in an XY model, the input file comes
#  from a simulation in the same folder and contains angles.
# It should also provide some help in plotting the XY system with nice arrows.

import matplotlib
import numpy as np
import argparse
import sys

matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
        "config_file", help="Raw spin configuration file")
parser.add_argument(
        "-ls", "--lattice_size",
        help="Linear size of the lattice (ex. 32)",
        type=int, required=True)
parser.add_argument(
        "-dg", "--draw_graph",
        help="Print graphs in PDF files",
        required=False, action='store_true')
args = parser.parse_args()


# lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]


def lol(lst, size):
    """Convert a list to matrix.
    """

    copy_lst = []
    for i in range(0, len(lst), size):
        copy_lst.append(lst[i:i+size])

    return np.array(copy_lst)


def saw(x):
    """Saw tooth function.
    """

    if x <= - np.pi:
        return x + 2 * np.pi
    elif - np.pi <= x <= np.pi:
        return x
    elif np.pi <= x:
        return x - 2 * np.pi


def column(matrix, i):
    """Return column row as list.
    """
    return [row[i] for row in matrix]


input_set = sys.argv[1]

L = args.lattice_size    # Size
s = 2     # Submatrix size
count = 0
v = []    # Vorticity vector

graph = args.draw_graph   # Set True if you want to create vortex graphics

with open(input_set, 'r') as infile, \
     open(str(input_set) + "_vortex", 'a') as outfile:
    for line in infile:
        if count % 2 == 0:
            T = line.split()[0]     # Store temperature
            outfile.write(str(T) + "\n")
            count += 1
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

            # Calculate vorticity for each sxs submatrix
            for l in range(0, L):
                for j in range(0, L):
                    wn = ((saw(h[l][j].item((0, 0)) - h[l][j].item((0, 1)))
                          + saw(h[l][j].item((0, 1)) - h[l][j].item((1, 1)))
                          + saw(h[l][j].item((1, 1)) - h[l][j].item((1, 0)))
                          + saw(h[l][j].item((1, 0)) - h[l][j].item((0, 0))))
                          / (2 * np.pi))
                    wn = np.round(wn, decimals=0)
                    wn = np.int8(wn)
                    v.append(wn)        # add to vorticity array

            outfile.write(str(" ".join(repr(e) for e in v)) + '\n')

            count += 1

            if graph:
                # Convert array in LxL matrix (work only for square)
                V = lol(v, L)
                V = np.array(V, dtype=float)

                # Plot configuration and save it on .pdf
                [X, Y] = np.mgrid[0:L, 0:L]
                X_plot = np.cos(XY)
                Y_plot = np.sin(XY)

                plt.figure()
                # ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
                # plt.quiver(X, Y, X_plot, Y_plot, color='blue', pivot='mid')
                plt.quiver(X, Y, X_plot, Y_plot, V, pivot='mid')
                plt.axis('equal')
                # plt.axis('off')
                cbar = plt.colorbar(ticks=[-1, 1])
                # cbar.ax.set_yticklabels(['-PI', '0', 'PI'])

                name = 'XY-T_' + str(T)
                figname = name + '.pdf'
                plt.savefig(figname, format='pdf', bbox_inches='tight')

                filename = name + '.data'
                np.savetxt(filename, XY)


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
