#!/usr/bin/env python

import numpy as np
import struct
import matplotlib.pyplot as plt

NCHAN = 4097
HDRSZ = 4096 # Bytes
DATSZ = 4    # Bytes

dada_filename = "../data/2020-01-21-01:01:01_0000000000000000.000000.dada"
dada_file = open(dada_filename, "rb")
dada_file.seek(HDRSZ)

binary_spec = dada_file.read()
#spec_cuda = 20*np.log10(np.array(struct.unpack('4097f', binary_spec)))
spec_cuda = np.array(struct.unpack('4097f', binary_spec))
dada_file.close()

print(spec_cuda[-1])

npy_filename = "../data/sat_samples_spec_8192_32bit.npy"
spec_py = 10**(np.array(np.load(npy_filename))/20.0)
#spec_py = np.load(npy_filename)

#print(len(spec_py))

spec_diff = spec_py - spec_cuda

plt.figure()
plt.subplot(2,1,1)
plt.plot(spec_diff/spec_py)
#plt.plot(spec_diff)
plt.subplot(2,1,2)
plt.plot(spec_py)
plt.plot(spec_cuda)
plt.show()

#npy_filename = "../data/sat_samples_spec_8192.npy"
#spec_py = 10**(np.array(np.load(npy_filename))/20.0)
#
#npy_filename1 = "../data/sat_samples_spec_8192_32bit.npy"
#spec_py1 = 10**(np.array(np.load(npy_filename))/20.0)
#
#spec_diff = spec_py - spec_py1
#
#plt.figure()
#plt.subplot(2,1,1)
#plt.plot(spec_diff/spec_py)
##plt.plot(spec_diff)
#plt.subplot(2,1,2)
#plt.plot(spec_py)
#plt.plot(spec_py1)
#plt.show()

