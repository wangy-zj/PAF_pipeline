#!/usr/bin/env python

import numpy as np
import struct
import matplotlib.pyplot as plt
import logging

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='A Python script to pad DADA header to 4096 bytes', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose',   action='store_true',      help='Be verbose')
    parser.add_argument('-f', '--file_name', default=None, type=str,   help='File name for input and output')
    
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    #print(f'Values={values}')

    file_name = values.file_name

    f = open(file_name, "r")
    header = f.read().ljust(4096,' ')
    print(header)
    f.close()
    
if __name__ == '__main__':
    _main()
