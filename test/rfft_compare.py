#!/usr/bin/env python

import numpy as np
import struct
import matplotlib.pyplot as plt


complex_data_py   = np.loadtxt("../data/fft_result_py_64.txt", dtype=np.complex_)

real_data_cuda = np.loadtxt("../data/fft_result_cuda_64.txt")

complex_data_cuda = real_data_cuda[:,0] + 1j*real_data_cuda[:,1]

diff_data = complex_data_py - complex_data_cuda

diff_data_x = diff_data.real
diff_data_y = diff_data.imag

power_py   = np.absolute(complex_data_py)**2
power_cuda = np.absolute(complex_data_cuda)**2

diff_power = power_py - power_cuda

#print(complex_data_cuda)

print(diff_data)

power_sum_py = np.sum(power_py)
power_sum_cuda = np.sum(power_cuda)
print(power_sum_py)
print(power_sum_cuda)

print(power_sum_py/power_sum_cuda)

#plt.figure()
#plt.plot(power_py)
#plt.show()
#
##plt.figure()
##plt.subplot(2,1,1)
##plt.plot(diff_data_x)
###plt.plot(complex_data_py.real)
##
##plt.subplot(2,1,2)
##plt.plot(diff_data_y)
###plt.plot(complex_data_py.imag)
##
##plt.show()
#
#plt.figure()
#plt.subplot(3,1,1)
#plt.plot(diff_power/power_py)
#plt.subplot(3,1,2)
#plt.plot(diff_data_x/complex_data_py.real)
#plt.subplot(3,1,3)
#plt.plot(diff_data_y/complex_data_py.imag)
#plt.show()
