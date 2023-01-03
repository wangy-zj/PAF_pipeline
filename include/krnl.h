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

/*!
  The input data is 1d time series, which has only one antenna and one polarisation
  input data is 8-bits signed int and output should be 32-bits float
  * @param[in]  input    Input raw data in int8_t
  * @param[in]  output   Output unpacked data as 32-bits float
  * @param[in]  nsamp    Number of samples

  nsamp is defined by gridDim.x and blockDim.x
  where blockDim.x is configured as nthread 
  gridDim.x = nsamp/nthread
  
 */
__global__ void krnl_unpack_1ant1pol(int8_t *input, float *output, int nsamp);

/*!
  As we are doing real to complex FFT, NCHAN is NPOINT/2+1, where NPOINT is points of real to complex FFT

  The input data is FFTed complex data with order in [NFFT NCHAN], where NFFT is nsamp/NPOINT
  in the between, we power complex data to 32-bit float
  The output data is accumulated in time and only NCHAN data left

  nchan is defined by blockDim.x and gridDim.x, where blockDim.x is nthread
  nfft is defined by gridDim.y
  ideally we need to transport data from [NFFT NCHAN] to [NCHAN NFFT] before we do time accumulation
*/
__global__ void krnl_unpack(int8_t *input, cuComplex *output, int nsamp, int inter,int chan);

__global__ void krnl_power_beamform_1ant1pol(cuComplex *input, float *output, int reset);

__global__ void krnl_power_taccumulate_1ant1pol(cuComplex *input, float *output, int nfft, int nchan, int reset);

/*!
  This kernel is purely for the transpose of [NFFT-NCHAN] data into [NCHAN-NFFT]
  n: NCHAN
  m: NFFT

  gridDim.x = ceil(NCHAN / TILE_DIM) = ceil(n / TILE_DIM);
  gridDim.y = ceil(NFFT  / TILE_DIM) = ceil(m / TILE_DIM);
  gridDim.z = 1;
  blockDim.x = TILE_DIM;
  blockDim.y = NROWBLOCK_TRANS;
  blockDim.z = 1;
  
  Load matrix into tile
  Every Thread loads in this case 4 elements into tile.
  TILE_DIM/NROWBLOCK_TRANS = 32/8 = 4 
  
*/
__global__ void krnl_tf2ft_1ant1pol(const cuComplex* in, cuComplex *out, int m, int n);
  
#endif