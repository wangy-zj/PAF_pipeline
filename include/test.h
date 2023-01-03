#ifndef _TEST_H
#define _TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/time.h>
#include <getopt.h>
#include <inttypes.h>

#include <complex>

#ifdef __cplusplus
extern "C" {
#endif
  
#define CPP_COMPLEX std::complex<float>

  int unpack_1ant1pol(int8_t *input, float *output, int nsamp);
  int power_taccumulate_1ant1pol(CPP_COMPLEX *input, float *output, int nfft, int nchan, int reset);
  int tf2ft_1ant1pol(CPP_COMPLEX *input, CPP_COMPLEX *output, int nfft, int nchan);

#ifdef __cplusplus
}
#endif
  
#endif
