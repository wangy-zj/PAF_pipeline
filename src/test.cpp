#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../include/test.h"

int unpack_1ant1pol(int8_t *input, float *output, int nsamp){

  for(int i = 0; i < nsamp; i++){
    output[i] = (float)input[i];
  }
  
  return EXIT_SUCCESS;
}

int power_taccumulate_1ant1pol(CPP_COMPLEX *input, float *output, int nfft, int nchan, int reset){
  
  for(int ichan = 0; ichan < nchan; ichan++){
    float accumulate = 0;
    for(int ifft = 0; ifft < nfft; ifft++){
      int index = ifft*nchan + ichan;
      float amp = std::abs(input[index]);
      accumulate += (amp*amp);
    }
    
    if(reset){
      output[ichan] = accumulate;
    }
    else{
      output[ichan] += accumulate;
    }
  }
  return EXIT_SUCCESS;
}

int tf2ft_1ant1pol(CPP_COMPLEX *input, CPP_COMPLEX *output, int nfft, int nchan){
  for(int ifft = 0; ifft < nfft; ifft++){
    for(int ichan = 0; ichan < nchan; ichan++){
      int index_input  = ifft*nchan+ichan;
      int index_output = ichan*nfft+ifft;

      output[index_output] = input[index_input];
    }
  }

  return EXIT_SUCCESS;
}

