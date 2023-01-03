#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
  This is the main function to test krnl_tf2ft_1ant1pol kernel.
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

TEST_CASE("tf2ft_1ant1pol") {
  
  // Setup
  int gpu = 0;
  int npkt = 8192;
  int nthread = 128;
  int nfft_point = 8192;
  
  int nchan = nfft_point/2+1;
  int nfft  = npkt*NSAMP_PKT/nfft_point;
  int nsamp = nfft*nchan;
  
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
  float stddev = 10;
  curandGenerator_t gen;
  checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

  // data
  RealGeneratorNormal data_r(gen, mean, stddev, nsamp);
  RealGeneratorNormal data_i(gen, mean, stddev, nsamp);
  ComplexBuilder<float, float, cuComplex> data(data_r.data, data_i.data, nsamp, nthread);
  
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
  ManagedMemoryAllocator<CPP_COMPLEX> h_result(nsamp);
  tf2ft_1ant1pol((CPP_COMPLEX*)data.data, h_result.data, nfft, nchan);
  
#ifdef TIMEIT
  clock_gettime(CLOCK_REALTIME, &c_stop);
  c_elapsed = (c_stop.tv_sec - c_start.tv_sec) +
    (c_stop.tv_nsec - c_start.tv_nsec)/1.0E9L;

  fprintf(stdout, "tsamp_pkt is %f milliseconds, we have %d packets, available time is %f milliseconds\n", TSAMP_PKT, npkt, TSAMP_PKT*npkt);
  fprintf(stdout, "elapsed_time for CPU version is %f milliseconds\n", 1.0E3*c_elapsed);
#endif
  
  fprintf(stdout, "Finish CPU execution\n");
  
  // Now run kernel on GPU
  dim3 grid_size(ceil(nchan/(double)TILE_DIM), ceil(nfft/(double)TILE_DIM), 1);
  dim3 block_size(TILE_DIM, NROWBLOCK_TRANS, 1);

  ManagedMemoryAllocator<cuComplex> d_result(nsamp);
  print_cuda_memory_info();
  
#ifdef TIMEIT
  CUDA_STARTTIME(g);
#endif

  krnl_tf2ft_1ant1pol<<<grid_size, block_size>>>(data.data, d_result.data, nfft, nchan);
  getLastCudaError("Kernel execution failed [ krnl_tf2ft_1ant1pol ]");
  
#ifdef TIMEIT
  // CUDA_STOPTIME has sync inside
  CUDA_STOPTIME(g);
  fprintf(stdout, "elapsed time for krnl_tf2ft_1ant1pol running on %s is %f milliseconds\n\n", prop.name, gtime);
#else
  checkCudaErrors(cudaDeviceSynchronize());
#endif

  // Now check numbers as complex
  RealDifferentiator<float, float> diff((float*)d_result.data, (float*)d_result.data, 2*nsamp, nthread);
  
  RealMeanStddevCalculator<float> mean_stddev_diff(diff.data, 2*nsamp, nthread, 7);
  
  std::cout << "\n";
  std::cout << mean_stddev_diff.mean << "\t" << mean_stddev_diff.stddev << std::endl;
  std::cout << std::endl;

  // Free allocated memory
  checkCudaErrors(curandDestroyGenerator(gen));
}
