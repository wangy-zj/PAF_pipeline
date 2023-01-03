#!/usr/bin/env python

import h5py
import numpy as np
import struct
import matplotlib.pyplot as plt
import logging

# python test_rfft.py -c 4097 -f 128 -m 0 -s 64

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='A Python script to apply rfft and record data to compare with CUDA result', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose',   action='store_true',      help='Be verbose')
    parser.add_argument('-c', '--nchan',     default=None, type=int,   help='Number of fine channels')
    parser.add_argument('-f', '--nfft',      default=None, type=int,   help='Number of FFT to run')
    parser.add_argument('-m', '--mu',        default=None, type=float, help='Mean of gaussian random numbers')
    parser.add_argument('-s', '--sigma',     default=None, type=float, help='Sgima of gaussian random numbers')
    
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    print(f'Values={values}')

    nchan = values.nchan
    nfft  = values.nfft
    mu    = values.mu
    sigma = values.sigma
    
    fft_length  = (nchan-1)*2
    data_length = nfft*fft_length
    
    print(f"nchan       {nchan}")
    print(f"nfft        {nfft}")
    print(f"mu          {mu}")
    print(f"sigma       {sigma}")
    
    print(f"fft_length  {fft_length}")
    print(f"data_length {data_length}")

    random_data = np.random.normal(mu, sigma, data_length)
    np.savetxt("../data/input_random_64.txt", random_data)

    # now do fft
    fft_result = []
    for i in range(nfft):
        temp_result = np.fft.rfft(random_data[i*fft_length:(i+1)*fft_length])
        fft_result.append(temp_result)
    fft_result = np.array(fft_result).flatten()
    power_result = np.absolute(fft_result)**2
    
    np.savetxt("../data/fft_result_py_64.txt", fft_result)
    np.savetxt("../data/power_result_py_64.txt", power_result)
    
if __name__ == '__main__':
    _main()
