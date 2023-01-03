#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import time

# 1. CUDA FFT, raw 8-bits int with python np.fft.rfft, gives us the same error at RFI frequency
# 2. raw 8-bits int and 32-bits float unpack with CUDA, run with np.fft.rfft, gives the same error at RFI frequency
# 3. raw 8-bits int and 32-bits float unpack with CUDA, accumulation has small error, but accumulated result is outside of the range of 8-bits int
# 4. raw 8-bits int and the same data astype(np.float32), gives exactly the same power numbers, no error at any frequency

#fft_length = 8192
#
#raw_data_fname = "../data/sat_samples_mini.bin"
#raw_data       = np.fromfile(raw_data_fname, dtype=np.int8)
#
#cuda_fft_fname = "../data/fft.bin"
#cuda_fft       = np.fromfile(cuda_fft_fname, dtype=np.csingle)
#
#ndata = raw_data.shape[0]
#nfft  = ndata//fft_length
#nchan = fft_length//2+1
#
#raw_power  = np.zeros(nchan)
#cuda_power = np.zeros(nchan)
#
#for ifft in range(nfft):
#    raw_data_current = raw_data[ifft*fft_length: (ifft+1)*fft_length]
#    raw_fft  = np.fft.rfft(raw_data_current)
#
#    raw_power += (raw_fft.real**2 + raw_fft.imag**2)
#    
#    cuda_fft = cuda_fft[ifft*nchan: (ifft+1)*nchan]
#    cuda_power += (cuda_fft.real**2 + cuda_fft.imag**2)
#
#    print(f'{ifft+1} vs {nfft}, {100*(ifft+1)/float(nfft):.3f}% done')
#    
#diff_power = raw_power - cuda_power
#
#plt.figure()
#plt.plot(diff_power/raw_power)
#plt.show()
#
#plt.figure()
#plt.plot(diff_power/cuda_power)
#plt.show()

#fft_length = 8192
#
#raw_data_fname = "../data/sat_samples_mini.bin"
#raw_data       = np.fromfile(raw_data_fname, dtype=np.int8)
#
#unpack_data_fname = "../data/unpack.bin"
#unpack_data       = np.fromfile(unpack_data_fname, dtype=np.float32)
#
#ndata = raw_data.shape[0]
#nfft  = ndata//fft_length
#nchan = fft_length//2+1
#
#raw_power    = np.zeros(nchan)
#unpack_power = np.zeros(nchan)
#
#for ifft in range(nfft):
#    raw_data_current = raw_data[ifft*fft_length: (ifft+1)*fft_length]
#    raw_fft  = np.fft.rfft(raw_data_current)
#
#    raw_power += (raw_fft.real**2 + raw_fft.imag**2)
#    
#    unpack_data_current = unpack_data[ifft*fft_length: (ifft+1)*fft_length]
#    unpack_fft  = np.fft.rfft(unpack_data_current)
#    
#    unpack_power += (unpack_fft.real**2 + unpack_fft.imag**2)
#
#    print(f'{ifft+1} vs {nfft}, {100*(ifft+1)/float(nfft):.3f}% done')
#    
#diff_power = raw_power - unpack_power
#
#plt.figure()
#plt.plot(diff_power/raw_power)
#plt.show()
#
#plt.figure()
#plt.plot(diff_power/unpack_power)
#plt.show()

#fft_length = 8192
#
#raw_data_fname = "../data/sat_samples_mini.bin"
#raw_data       = np.fromfile(raw_data_fname, dtype=np.int8)
#
#unpack_data_fname = "../data/unpack.bin"
#unpack_data       = np.fromfile(unpack_data_fname, dtype=np.float32)
#
#ndata = raw_data.shape[0]
#nfft  = ndata//fft_length
#nchan = fft_length//2+1
#
#raw    = np.zeros(fft_length)
#unpack = np.zeros(fft_length)
#
#for ifft in range(nfft):
#    raw    += raw_data[ifft*fft_length: (ifft+1)*fft_length]
#    unpack += unpack_data[ifft*fft_length: (ifft+1)*fft_length]
#
#    print(f'{ifft+1} vs {nfft}, {100*(ifft+1)/float(nfft):7.3f}% done')
#diff = raw - unpack
#
#plt.figure()
#plt.plot(diff)
#plt.plot(raw/float(nfft))
#plt.plot(unpack/float(nfft))
#plt.show()
#
#plt.figure()
#plt.plot(diff/raw)
#plt.show()
#
#plt.figure()
#plt.plot(diff/unpack)
#plt.show()
#

fft_length = 8192

raw_data_fname = "../data/sat_samples_mini.bin"
raw_data       = np.fromfile(raw_data_fname, dtype=np.int8)
raw_data_float = raw_data.copy().astype(np.float32)

print(f'raw_data data type is {type(raw_data[0])}')
print(f'raw_data_float data type is {type(raw_data_float[0])}')

ndata = raw_data.shape[0]
nfft  = ndata//fft_length
nchan = fft_length//2+1

raw_power       = np.zeros(nchan)
raw_power_float = np.zeros(nchan)

for ifft in range(nfft):
    raw_data_current       = raw_data[ifft*fft_length: (ifft+1)*fft_length]
    raw_data_float_current = raw_data_float[ifft*fft_length: (ifft+1)*fft_length]

    raw_fft       = np.fft.rfft(raw_data_current)
    raw_fft_float = np.fft.rfft(raw_data_float_current)

    raw_power += (raw_fft.real**2 + raw_fft.imag**2)
    raw_power_float += (raw_fft_float.real**2 + raw_fft_float.imag**2)
    
    print(f'{ifft+1} vs {nfft}, {100*(ifft+1)/float(nfft):7.3f}% done')
    
diff_power = raw_power - raw_power_float

plt.figure()
plt.plot(diff_power/raw_power)
plt.show()

plt.figure()
plt.plot(diff_power/raw_power_float)
plt.show()
