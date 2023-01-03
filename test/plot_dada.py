#!/usr/bin/env python

import numpy as np
import struct
import matplotlib.pyplot as plt

NCHAN = 4097
HDRSZ = 4096 # Bytes
DATSZ = 4    # Bytes

dada_filename = "2020-01-21-01:01:01_0000000000000000.000000.dada"
dada_file = open(dada_filename, "rb")
dada_file.seek(HDRSZ)

binary_data = dada_file.read()
ntime = len(binary_data)//(NCHAN*DATSZ)

data = np.array(struct.unpack('4097f', binary_data))
#data = data.reshape((ntime, int(NCHAN)))

plt.figure()
#plt.plot(20*np.log10(((NCHAN-1)*2)**2*data)[1:])
plt.plot(20*np.log10(data)[1:])
#plt.plot(20*np.log10(data)[1:])
#plt.plot(data)
plt.show()

dada_file.close()
