#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../include/cuda/cuda_utilities.h"
#include "../include/krnl.h"

#define STRLEN 1024

__global__ void krnl_power(cuComplex *input, float *output, int nsamp){

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if(index<nsamp){
    float amp = cuCabsf(input[index]);
    output[index] = amp*amp;
  }
}

int main(int argc, char *argv[]){

  int gpu = 0;
  int nchan = 4097;
  int fft_length = (nchan-1)*2;
  int nfft = 128;
  int input_length = nfft*fft_length;
  int output_length = nfft*nchan;
  int nthread = 128;

  int nblock = output_length/nthread+1;
  
  char random_fname[STRLEN] = {0};
  char fft_fname[STRLEN] = {0};
  char power_fname[STRLEN] = {0};

  sprintf(random_fname, "../data/input_random_64.txt");
  sprintf(fft_fname, "../data/fft_result_cuda_64.txt");
  sprintf(power_fname, "../data/power_result_cuda_64.txt");
    
  fprintf(stdout, "DEBUG: gpu = %d\n", gpu);
  fprintf(stdout, "DEBUG: nfft = %d\n", nfft);
  fprintf(stdout, "DEBUG: nchan = %d\n", nchan);
  fprintf(stdout, "DEBUG: fft_length = %d\n", fft_length);
  fprintf(stdout, "DEBUG: input_length = %d\n", input_length);
  fprintf(stdout, "DEBUG: nthread = %d\n", nthread);

  FILE *random_file = fopen(random_fname, "r");
  FILE *fft_file = fopen(fft_fname, "w");
  FILE *power_file = fopen(power_fname, "w");
  
  // Setup GPU with ID and print out its name
  cudaDeviceProp prop = {0};
  int gpu_get = gpuDeviceInit(gpu); // The required gpu might be different from what we get
  fprintf(stdout, "Asked for GPU %d, got GPU %d\n", gpu, gpu_get);
  checkCudaErrors(cudaGetDeviceProperties(&prop, gpu_get));
  fprintf(stdout, "GPU name is %s\n", prop.name);

  // Setup FFT
  cufftHandle fft_plan;
  checkCudaErrors(cufftPlan1d(&fft_plan, fft_length, CUFFT_R2C, nfft));

  float *input = NULL;
  cuComplex *output = NULL;
  float *power = NULL;
  
  checkCudaErrors(cudaMallocManaged(&input, input_length*sizeof(float), cudaMemAttachGlobal));
  checkCudaErrors(cudaMallocManaged(&output, output_length*sizeof(cuComplex), cudaMemAttachGlobal));
  checkCudaErrors(cudaMallocManaged(&power, output_length*sizeof(float), cudaMemAttachGlobal));

  // Read in input data
  char line[STRLEN] = {0};
  for (int i = 0; i < input_length; i++){
    fgets(line, STRLEN, random_file);
    sscanf(line, "%f", &input[i]);
    //fprintf(stdout, "%f\n", input[i]);
  }
  fprintf(stdout, "HERE\n");
  
  checkCudaErrors(cufftExecR2C(fft_plan, input, output));
  fprintf(stdout, "HERE\n");

  krnl_power<<<nblock, nthread>>>(output, power, output_length);
  getLastCudaError("Kernel execution failed [ krnl_power ]");

  // For managed memory, we need to sync before read data from device
  checkCudaErrors(cudaDeviceSynchronize());
  
  // write out result
  for (int i = 0; i < output_length; i++){
    fprintf(fft_file, "%.18E %.18E\n", output[i].x, output[i].y);
    fprintf(power_file, "%.18E\n", power[i]);
  }
  
  // Free memory
  checkCudaErrors(cudaFree(input));
  checkCudaErrors(cudaFree(output));
  checkCudaErrors(cudaFree(power));
  
  fprintf(stdout, "HERE\n");
  
  // close files
  fclose(random_file);
  fclose(fft_file);
  fclose(power_file);
  fprintf(stdout, "HERE\n");
  
  return EXIT_SUCCESS;
}
