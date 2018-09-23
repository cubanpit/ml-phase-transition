#!/usr/bin/python3

# Machine Learning programm written using TensorFlow
# Data used to train the neural network come from a computer simulated 2D Ising
#  model, the purpose is to identify critical phase transitions using a trained
#  neural network, without feeding it with the order parameter.

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
        honeycomb (hc), cubic (cb)", required=True)
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
                if temperature < critical_temp:
                    binary_temperatures.append(np.array([1, 0]))
                else:
                    binary_temperatures.append(np.array([0, 1]))
                odd = False

            else:
                configuration = np.fromstring(line, dtype=np.int8, sep=' ')
                configurations.append(configuration)
                odd = True

    magnetizations = np.array(magnetizations).astype(np.float32)
    binary_temperatures = np.array(binary_temperatures).astype(np.uint8)
    real_temperatures = np.array(real_temperatures).astype(np.float32)
    configurations = np.array(configurations).astype(np.int8)

    return magnetizations, binary_temperatures, real_temperatures, configurations


def critical_temp(input_lattice):
    """Returns critical temperature for different lattice.
    """

    square_temp = 2/np.log(1+np.sqrt(2))
    triangular_temp = 4/np.log(3)
    cubic_temp = 1/0.221654
    honeycomb_temp = 1/0.658478

    if input_lattice == "sq":
        test_temp = square_temp
    elif input_lattice == "tr":
        test_temp = triangular_temp
    elif input_lattice == "hc":
        test_temp = honeycomb_temp
    elif input_lattice == "cb":
        test_temp = cubic_temp
    else:
        raise SyntaxError("Use sq for square, tr for triangular and cb for cubic")

    return test_temp


def unique_elements(complete_array):
    """Returns a list of different elements in an array.
    """

    uniques = []
    for elem in complete_array:
        if elem not in uniques:
            uniques.append(elem)

    uniques = np.array(uniques)
    uniques.sort()

    return uniques


def unison_shuffled_copies(a, b):
    """Shuffle two arrays with corresponding elements.
        High memory usage, makes entire copy of arrays.
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



#
#   MAIN
#

# set test critical temperature based on lattice type
test_temp = critical_temp(args.lattice_type)

# load test set
test_set = args.test_set
test_magns, test_bin_temps, test_real_temps, test_configs \
        = read_data(test_set, test_temp)

test_configs = test_configs[:10000]
test_real_temps = test_real_temps[:10000]

x_data = np.asarray(test_configs).astype('float64')
print("Data shape =", x_data.shape)

print("Converting with t-sne...")
vis_data = TSNE(n_iter=5000, perplexity=100, verbose=1).fit_transform(x_data)

print("Plotting data...")
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]
plt.scatter(vis_x, vis_y, c=test_real_temps, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar()
plt.show()

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
