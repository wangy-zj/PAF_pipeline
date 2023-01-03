#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
  This is the main function to test krnl_power_taccumulate_1ant1pol kernel.
  the kernel does not produce dig value?
*/

#include "../include/cuda/cuda_utilities.h"
#include "../include/test.h"
#include "../include/krnl.h"

#include <iostream> // For cout

#define TSAMP_PKT 8.192E-3 // milliseconds
#define NSAMP_PKT 8192

#define TIMEIT

#ifdef TIMEIT
#define CUDA_STARTTIME(x)  cudaEventRecord(x ## _start, 0);

#define CUDA_STOPTIME(x) {					\
    float dtime;						\
    cudaEventRecord(x ## _stop, 0);				\
    cudaEventSynchronize(x ## _stop);				\
    cudaEventElapsedTime(&dtime, x ## _start, x ## _stop);	\
    x ## time += dtime; }

#else
#define CUDA_STARTTIME(x)
// It is better to sync even we do not timing information
#define CUDA_STOPTIME(x)
#endif

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../doctest/doctest.h"

TEST_CASE("power_taccumulate_1ant1pol") {
  
  // Setup
  int gpu = 0;
  //int npkt = 8192;
  int npkt = 8192;
  int nthread = 128;
  int nfft_point = 8192;
  
  int nchan = nfft_point/2+1;
  int nfft  = npkt*NSAMP_PKT/nfft_point;
  int ninput = nfft*nchan;

  int reset = 1;
  
  fprintf(stdout, "DEBUG: gpu = %d\n", gpu);
  fprintf(stdout, "DEBUG: npkt = %d\n", npkt);
  fprintf(stdout, "DEBUG: nfft = %d\n", nfft);
  fprintf(stdout, "DEBUG: nchan = %d\n", nchan);
  fprintf(stdout, "DEBUG: nthread = %d\n", nthread);
  
  // Setup GPU with ID and print out its name
  cudaDeviceProp prop = {0};
  int gpu_get = gpuDeviceInit(gpu); // The required gpu might be different from what we get
  fprintf(stdout, "Asked for GPU %d, got GPU %d\n", gpu, gpu_get);
  checkCudaErrors(cudaGetDeviceProperties(&prop, gpu_get));
  fprintf(stdout, "GPU name is %s\n", prop.name);

  // Setup buffers  
  float mean = 0;
  float stddev = 6400;
  curandGenerator_t gen;
  checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

  // data
  RealGeneratorNormal data_r(gen, mean, stddev, ninput);
  RealGeneratorNormal data_i(gen, mean, stddev, ninput);
  ComplexBuilder<float, float, cuComplex> data(data_r.data, data_i.data, ninput, nthread);
  
  // Setup timer for both CPU and GPU
#ifdef TIMEIT
  // Setup CPU timer
  struct timespec c_start = {0};
  struct timespec c_stop  = {0};
  double c_elapsed = 0;
  
  // Setup GPU timer
  cudaEvent_t g_start = {0};
  cudaEvent_t g_stop  = {0};
  float gtime = 0;
  checkCudaErrors(cudaEventCreate(&g_start));
  checkCudaErrors(cudaEventCreate(&g_stop));
  clock_gettime(CLOCK_REALTIME, &c_start);
#endif
  
  // Do processing on CPU
  ManagedMemoryAllocator<float> h_result(nchan);
  power_taccumulate_1ant1pol((CPP_COMPLEX*)data.data, h_result.data, nfft, nchan, reset);
  
#ifdef TIMEIT
  clock_gettime(CLOCK_REALTIME, &c_stop);
  c_elapsed = (c_stop.tv_sec - c_start.tv_sec) +
    (c_stop.tv_nsec - c_start.tv_nsec)/1.0E9L;

  fprintf(stdout, "tsamp_pkt is %f milliseconds, we have %d packets, available time is %f milliseconds\n", TSAMP_PKT, npkt, TSAMP_PKT*npkt);
  fprintf(stdout, "elapsed_time for CPU version is %f milliseconds\n", 1.0E3*c_elapsed);
#endif
  
  fprintf(stdout, "Finish CPU execution\n");
  
  // Now run kernel on GPU
  ManagedMemoryAllocator<float> d_result(nchan);
  
#ifdef TIMEIT
  CUDA_STARTTIME(g);
#endif

  krnl_power_taccumulate_1ant1pol<<<nchan/nthread+1, nthread>>>(data.data, d_result.data, nfft, nchan, reset);
  getLastCudaError("Kernel execution failed [ krnl_power_taccumulate_1ant1pol ]");
  
#ifdef TIMEIT
  // CUDA_STOPTIME has sync inside
  CUDA_STOPTIME(g);
  fprintf(stdout, "elapsed time for krnl_power_taccumulate_1ant1pol running on %s is %f milliseconds\n\n", prop.name, gtime);
#else
  checkCudaErrors(cudaDeviceSynchronize());
#endif

  // Now check numbers as complex
  RealDifferentiator<float, float> diff(d_result.data, h_result.data, nchan, nthread);
  
  RealMeanStddevCalculator<float> mean_stddev_diff(diff.data, nchan, nthread, 7);

  for(int i = 0; i < nchan; i++){
    //if(h_result.data[i]!=h_result.data[i])
      //std::cout << i << "\t" << h_result.data[i] << "\t" << h_result.data[i] << std::endl;
      fprintf(stdout, "%d\t%f\t%f\t%E\n", i, h_result.data[i], d_result.data[i], (h_result.data[i]-d_result.data[i])/h_result.data[i]);
  }
  std::cout << "\n";
  std::cout << mean_stddev_diff.mean << "\t" << mean_stddev_diff.stddev << std::endl;
  std::cout << std::endl;

  // Free allocated memory
  checkCudaErrors(curandDestroyGenerator(gen));
}
