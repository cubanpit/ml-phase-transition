#!/usr/bin/python3

# File: view_data_tsne.py

# Simple program to draw t-SNE representation of spin configurations for
#  Ising and XY model. Spin configurations come from a MonteCarlo simulation.

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "test_set", help="Test set file")
parser.add_argument(
        "-lt", "--lattice_type",
        help="Test set lattice type: square (sq), triangular (tr),\
        honeycomb (hc), cubic (cb), xy", required=True)
parser.add_argument(
        "-dn", "--data_number",
        help="Number of data to analyze and plot",
        type=int, default=1000, required=False)
args = parser.parse_args()


def read_data(input_set, critical_temp):
    """Read data from file.

    Only argument is the path to the data file.

    File format:
    - odd lines contain magnetization and temperature separated by spaces
    - even lines contain spin configuration, single spin separated by spaces
    """

    magnetizations = []
    binary_temperatures = []
    real_temperatures = []
    configurations = []
    odd = True

    with open(input_set, 'r') as infile:
        for line in infile:
            if odd is True:
                infos = line.split()
                if len(infos) == 2:
                    magnetization, temperature = infos[:]
                elif len(infos) == 1:
                    temperature = infos[0]
                    magnetization = 0.0
                else:
                    raise RuntimeError(
                            "Wrong number of information on the same line.\n"
                            "Expected informations: (magnetization) "
                            "temperature")
                temperature = float(temperature)
                real_temperatures.append(temperature)
                magnetizations.append(magnetization)
                if temperature < critical_temp:
                    binary_temperatures.append(np.array([1, 0]))
                else:
                    binary_temperatures.append(np.array([0, 1]))
                odd = False

            else:
                configuration = np.fromstring(line, dtype=np.float32, sep=' ')
                configurations.append(configuration)
                odd = True

    magnetizations = np.array(magnetizations).astype(np.float32)
    binary_temperatures = np.array(binary_temperatures).astype(np.uint8)
    real_temperatures = np.array(real_temperatures).astype(np.float32)
    configurations = np.array(configurations).astype(np.float32)

    return magnetizations, binary_temperatures, \
           real_temperatures, configurations


def critical_temp(input_lattice):
    """Returns critical temperature for different lattice.
    """

    square_temp = 2/np.log(1+np.sqrt(2))
    triangular_temp = 4/np.log(3)
    cubic_temp = 1/0.221654
    honeycomb_temp = 1/0.658478
    xy_temp = 0.893

    if input_lattice == "sq":
        test_temp = square_temp
    elif input_lattice == "tr":
        test_temp = triangular_temp
    elif input_lattice == "hc":
        test_temp = honeycomb_temp
    elif input_lattice == "cb":
        test_temp = cubic_temp
    elif input_lattice == "xy":
        test_temp = xy_temp
    else:
        raise SyntaxError("Use sq for square, tr for triangular",
                          "and cb for cubic")

    return test_temp


#
#   MAIN
#

# set test critical temperature based on lattice type
test_temp = critical_temp(args.lattice_type)

# load test set
test_set = args.test_set
test_magns, test_bin_temps, test_real_temps, test_configs \
        = read_data(test_set, test_temp)

test_configs = test_configs[:args.data_number]
test_real_temps = test_real_temps[:args.data_number]

x_data = np.asarray(test_configs).astype('float64')

vis_data = TSNE(verbose=0).fit_transform(x_data)

vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

# plt.scatter(
#             vis_x, vis_y, c=test_real_temps,
#             cmap=plt.cm.get_cmap("jet", 40), s=10)
# plt.colorbar(orientation="horizontal")
# plt.show()

print("x y temp")
for d in range(len(vis_data)):
    print(vis_x[d], vis_y[d], test_real_temps[d])


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
