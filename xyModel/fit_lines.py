#! /bin/python3

# Fit points from file on a line and plot stuff

import sys
import numpy as np
import matplotlib.pyplot as plt

# receive data file as cli argument
data_file = sys.argv[1]

sizes = []
temps = []
temps_e = []
with open(data_file, 'r') as input_file:
    count = 0
    for line in input_file:
        # do not read first line
        if count is 0:
            count += 1
            continue
        if count is 1:
            count += 1
            tc_theo = float(line)
        else:
            data = line.split()
            sizes.append(int(data[0]))
            temps.append(float(data[1]))
            temps_e.append(float(data[2]))

sizes = np.array(sizes)
temps = np.array(temps)
temps_e = np.array(temps_e)

print("Lattice sizes :", sizes)
print("Temperatures :", temps)
print("Temperature errors :", temps_e)
print("Theoretical critical temperature :", tc_theo)

inv_sizes = np.array(1 / np.power(np.log(sizes), 2))
weights = np.array(1 / temps_e)
coeff, cov = np.polyfit(inv_sizes, temps, deg=1, w=weights, cov=True)
line = np.poly1d(coeff)
print(
        "Predicted critical temperature : ",
        np.round(line[0], decimals=4), " +- ",
        np.round(np.sqrt(cov[0,0]), decimals=4))
plt.errorbar(inv_sizes, temps, yerr=temps_e, linestyle='', marker='o')
plt.hlines(y=tc_theo, xmin=0, xmax=(np.max(inv_sizes)), color='r', label="critical temperature")
plt.plot(inv_sizes, line(inv_sizes), '-', color='y', label="data fit line")
plt.xlabel('1/ln(L)^2')
plt.ylabel('Temperature')
plt.legend(loc=2)
plt.grid(linestyle='dashed')
plt.show()
