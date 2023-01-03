#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../include/cuda/cuda_utilities.h"
#include "../include/krnl.h"

//#define TSAMP_PKT 8E-3 // milliseconds
//#define NSAMP_PKT 8192

#define TSAMP_PKT 4E-3 // milliseconds
#define NSAMP_PKT 4096

#define CUDA_STARTTIME(x)  cudaEventRecord(x ## _start, 0);

#define CUDA_STOPTIME(x) {					\
    float dtime;						\
    cudaEventRecord(x ## _stop, 0);				\
    cudaEventSynchronize(x ## _stop);				\
    cudaEventElapsedTime(&dtime, x ## _start, x ## _stop);	\
    x ## time += dtime; }

int main(int argc, char *argv[]){

  int npkt = 8192;
  int gpu = 0;
  int nthread = 128;
  int nfft_point = 8192;

  int nchan = nfft_point/2+1; // it is also output size
  int nsamp_packed = npkt*NSAMP_PKT;
  int nfft  = nsamp_packed/nfft_point;

  int nsamp_fft = nfft*nchan;
  int nrepeat = 1000;
  
  fprintf(stdout, "DEBUG: gpu = %d\n", gpu);
  fprintf(stdout, "DEBUG: npkt = %d\n", npkt);
  fprintf(stdout, "DEBUG: nfft = %d\n", nfft);
  fprintf(stdout, "DEBUG: nchan = %d\n", nchan);
  fprintf(stdout, "DEBUG: nthread = %d\n", nthread);
  fprintf(stdout, "DEBUG: nsamp_packed = %d\n", nsamp_packed);
  fprintf(stdout, "DEBUG: nsamp_fft = %d\n", nsamp_fft);

  // Setup GPU with ID and print out its name
  cudaDeviceProp prop = {0};
  int gpu_get = gpuDeviceInit(gpu); // The required gpu might be different from what we get
  fprintf(stdout, "Asked for GPU %d, got GPU %d\n", gpu, gpu_get);
  checkCudaErrors(cudaGetDeviceProperties(&prop, gpu_get));
  fprintf(stdout, "GPU name is %s\n", prop.name);

  // get packed data ready on device memory
  float mean = 0;
  float stddev = 10;
  curandGenerator_t gen;
  checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

  RealGeneratorNormal random_float(gen, mean, stddev, nsamp_packed);
  RealConvertor<float, int8_t> random(random_float.data, nsamp_packed, nthread);
  
  int8_t *h_packed = NULL;
  int8_t *d_packed = NULL;
  cuComplex *d_ffted = NULL;
  float *d_unpacked = NULL;
  float *d_result = NULL;
  float *h_result = NULL;
  
  checkCudaErrors(cudaMalloc(&d_packed, nsamp_packed*sizeof(int8_t)));
  checkCudaErrors(cudaMallocHost(&h_packed, nsamp_packed*sizeof(int8_t)));
  
  checkCudaErrors(cudaMalloc(&d_unpacked, nsamp_packed*sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_ffted, nsamp_fft*sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_result, nchan*sizeof(float)));
  checkCudaErrors(cudaMallocHost(&h_result, nchan*sizeof(float)));

  // Get input data to host
  checkCudaErrors(cudaMemcpy(h_packed, random.data, nsamp_packed*sizeof(int8_t), cudaMemcpyDeviceToHost));
  
  // Setup GPU timer
  cudaEvent_t g_start = {0};
  cudaEvent_t g_stop  = {0};
  float gtime = 0;
  checkCudaErrors(cudaEventCreate(&g_start));
  checkCudaErrors(cudaEventCreate(&g_stop));

  // Setup kernel sizes
  dim3 grid_unpack(nsamp_packed/nthread+1);
  dim3 grid_taccumulate(nchan/nthread+1);

  int reset = 1;
  // Setup FFT
  cufftHandle fft_plan;
  checkCudaErrors(cufftPlan1d(&fft_plan, nfft_point, CUFFT_R2C, nfft));
  
  CUDA_STARTTIME(g);
  for(int i = 0; i < nrepeat; i++){
    checkCudaErrors(cudaMemcpy(d_packed, h_packed, nsamp_packed*sizeof(int8_t), cudaMemcpyHostToDevice));
    
    krnl_unpack_1ant1pol<<<grid_unpack, nthread>>>(d_packed, d_unpacked, nsamp_packed);
    getLastCudaError("Kernel execution failed [ krnl_unpack_1ant1pol ]");
    
    checkCudaErrors(cufftExecR2C(fft_plan, d_unpacked, d_ffted));
    
    krnl_power_taccumulate_1ant1pol<<<grid_taccumulate, nthread>>>(d_ffted, d_result, nfft, nchan, reset);
    getLastCudaError("Kernel execution failed [ krnl_power_taccumulate_1ant1pol ]");
    
    checkCudaErrors(cudaMemcpy(h_result, d_result, nchan*sizeof(float), cudaMemcpyDeviceToHost));
  }
  CUDA_STOPTIME(g);
  
  fprintf(stdout, "tsamp_pkt is %f milliseconds, we have %d packets, available time is %f milliseconds\n", TSAMP_PKT, npkt, TSAMP_PKT*npkt);
  fprintf(stdout, "elapsed time for pipeline_1ant1pol running on %s is %f milliseconds (%d average)\n\n", prop.name, gtime/(float)nrepeat, nrepeat);

  cudaFree(d_packed);
  cudaFreeHost(h_packed);
  cudaFree(d_unpacked);
  cudaFree(d_ffted);
  cudaFree(d_result);
  cudaFreeHost(h_result);
  
  return EXIT_SUCCESS;
}
