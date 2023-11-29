#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../include/krnl.h"
#include "../pipeline/beamform.hh"


__global__ void krnl_unpack(int8_t *input, cuComplex *output, int nsamp){
  const int n_antenna = gridDim.x;
  const int n_timestep = gridDim.y;
  const int n_chan = blockDim.x;

  const int antenna_idx = blockIdx.x;
  const int timestep_idx = blockIdx.y;
  const int chan_idx = threadIdx.x;

  const int input_idx = antenna_idx * n_timestep * n_chan + timestep_idx * n_chan + chan_idx;
  const int output_idx = chan_idx * n_antenna * n_timestep + timestep_idx * n_antenna + antenna_idx;

  if(input_idx<nsamp){
    output[output_idx] = make_cuFloatComplex((float)input[input_idx*2],(float)input[input_idx*2+1]);
  }
}

__global__ void krnl_power_beamform(cuComplex *input, float *output, int nsamp_accu, int naverage){
  const int n_batch = gridDim.x;
  const int n_freq = gridDim.y;
  const int n_beams = blockDim.x;

  const int batching_idx = blockIdx.x;
  const int freq_idx = blockIdx.y;
  const int beam_idx = threadIdx.x;

  __shared__ float shmem[N_BEAM];
	shmem[beam_idx] = 0;			// initialize shared memory to zero

	const int input_idx = freq_idx * n_batch * naverage * n_beams 	// deal with frequency component first
								+ batching_idx * naverage * n_beams  		// deal with the indexing for each output next
																		// Each thread starts on the first time-pol index 
																		// (b/c we iterate over all time-pol indicies in for loop below)
								+ beam_idx;							   	// start each thread on the first element of each beam

	const int output_idx = batching_idx * n_freq * n_beams + freq_idx * n_beams + beam_idx;

	#pragma unroll
	for (int i = input_idx; i < input_idx + naverage*n_beams; i += n_beams) {
		shmem[beam_idx] += cuCabsf(input[i]);
	}

	output[output_idx] = shmem[beam_idx]; // slowest to fastest indicies: freq, beam
}

__global__ void krnl_power_zoomfft(cuComplex *input, float *output, int nfft, int nchan, int reset){
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

