#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import time
import struct

fname = "../data/sat_samples_mini.bin"

data = np.fromfile(fname, dtype=np.int8)

fft_length = 8192

ndata = data.shape[0]
nfft  = ndata//fft_length

start_fft = 650

for i in range(100):
    fft_index = start_fft + i
    fft_data = np.array(struct.unpack(f'{fft_length}b', data[fft_index*fft_length: (fft_index+1)*fft_length]))

    spec = np.fft.rfft(fft_data)
    print(f"working on {fft_index}")

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(fft_data)
    plt.subplot(2,1,2)
    plt.plot(np.absolute(spec)**2)
    plt.show()
