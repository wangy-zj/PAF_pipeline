#!/usr/bin/env python

import numpy as np
import struct
import matplotlib.pyplot as plt

power_py   = np.loadtxt("../data/power_result_py_64.txt")
power_cuda = np.loadtxt("../data/power_result_cuda_64.txt")
diff_power = power_py - power_cuda

plt.figure()
plt.plot(diff_power/power_py)
plt.show()
