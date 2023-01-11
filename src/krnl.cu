#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../include/krnl.h"


__global__ void krnl_unpack(int8_t *input, cuComplex *output, int nsamp, int inter,int chan){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<nsamp && index%8==0){
    output[index] = make_cuFloatComplex((float)input[index],(float)input[index+1]);
  }
}

__global__ void krnl_power_beamform(cuComplex *input, float *output, int reset){
  int block_index = blockIdx.x* blockDim.x;
  int index = blockIdx.x* blockDim.x + threadIdx.x;
  float amp = cuCabsf(input[index]);
  __syncthreads();
  if(reset){
    output[block_index] = 0;
  }else{
    output[block_index] += amp*amp;
  }
}

__global__ void krnl_power_zoomfft(cuComplex *input, float *output){

  //int isamp = blockIdx.x;
  int index = blockIdx.x* blockDim.x + threadIdx.x;
  int amp = cuCabsf(input[index]);
  output[index] = amp*amp;
}


__global__ void krnl_power_taccumulate_1ant1pol(cuComplex *input, float *output, int nfft, int nchan, int reset){

  int ichan = blockIdx.x*blockDim.x + threadIdx.x;
  if (ichan < nchan){
    float accumulate = 0;
    
    for(int ifft = 0; ifft < nfft; ifft++){
      int index = ifft*nchan + ichan;
      float amp = cuCabsf(input[index]);
      accumulate += (amp*amp); // power, not amp
    }

    // looks like numpy.fft.rfft has the same scaling factor as R2C cufft 
    if(reset){
      output[ichan] = accumulate;
    }else{
      output[ichan] += accumulate;
    }
  }
}

/*
  This kernel is purely for the transpose of [NFFT-NCHAN] data into [NCHAN-NFFT]
  n: NCHAN
  m: NFFT
*/
__global__ void krnl_tf2ft_1ant1pol(const cuComplex* in, cuComplex *out, int m, int n){  
  
  __shared__ cuComplex tile[TILE_DIM][TILE_DIM + 1];

  //gridsize_transpose.x = ceil(NCHAN / TILE_DIM) = ceil(n / TILE_DIM);
  //gridsize_transpose.y = ceil(NFFT  / TILE_DIM) = ceil(m / TILE_DIM);
  //gridsize_transpose.z = 1;
  //blocksize_transpose.x = TILE_DIM;
  //blocksize_transpose.y = NROWBLOCK_TRANS;
  //blocksize_transpose.z = 1;
  
  // Load matrix into tile
  // Every Thread loads in this case 4 elements into tile.
  // TILE_DIM/NROWBLOCK_TRANS = 32/8 = 4 
  
  int i_n = blockIdx.x * TILE_DIM + threadIdx.x;
  int i_m = blockIdx.y * TILE_DIM + threadIdx.y; 
  for (int i = 0; i < TILE_DIM; i += NROWBLOCK_TRANS) {
    if(i_n < n  && (i_m+i) < m){
      int64_t loc_in = (i_m+i)*n + i_n;
      tile[threadIdx.y+i][threadIdx.x] = in[loc_in];
    }
  }
  __syncthreads();
  
  i_n = blockIdx.y * TILE_DIM + threadIdx.x; 
  i_m = blockIdx.x * TILE_DIM + threadIdx.y;
  for (int i = 0; i < TILE_DIM; i += NROWBLOCK_TRANS) {
    if(i_n < m  && (i_m+i) < n){
      int64_t loc_out = (i_m+i)*m + i_n;
      out[loc_out] = tile[threadIdx.x][threadIdx.y+i];
    }
  }
}
