#ifndef _KRNL_H
#define _KRNL_H

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/time.h>
#include <getopt.h>
#include <inttypes.h>

#include <cuComplex.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>

#define NROWBLOCK_TRANS 8
#define TILE_DIM        32

/* unpack the DADA block from 8+8bit into cuComplex*/
__global__ void krnl_unpack(int8_t *input, cuComplex *output, int nsamp);

/* calculate the beamform power and integrate by n_average for each channel and beam*/
__global__ void krnl_power_beamform(cuComplex *input, float *output, int nsamp_acc, int naverage, int reset);

__global__ void krnl_power_zoomfft(cuComplex *input, float *output, int nfft, int nchan, int reset);
  
#endif