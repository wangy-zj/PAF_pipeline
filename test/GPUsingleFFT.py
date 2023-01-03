#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import time

fname = "../data/sat_samples_mini.bin"
data = np.fromfile(fname, dtype=np.int8)

data_length = data.shape[0]
fft_length  = 8192

nfft = data_length//fft_length

print (f"Len of data: {data_length} samples")
print (f"Len of FFT:  {fft_length} samples")
print (f"Number of FFT: {nfft}")

spec = np.zeros(fft_length//2+1)
for i in range(nfft//250):
    spec += np.absolute(np.fft.rfft(data[i*fft_length :(i+1)*fft_length]))**2
    #plt.plot(20*np.log10(spec))

plt.figure()
plt.plot(spec)
plt.show()
