#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
  This is the main function to check the performance of process
*/

#include <getopt.h>
#include <iostream>
#include <cublas_v2.h>
#include <cublas_api.h>

#include "../include/cuda/cuda_utilities.h"
#include "../include/test.h"
#include "../include/krnl.h"
#include "../include/dada_header.h"
#include "../include/dada_util.h"

#include "dada_cuda.h"
#include "beamform.hh"

#define CUDA_STARTTIME(x)  cudaEventRecord(x ## _start, 0);

#define CUDA_STOPTIME(x) {					\
    float dtime;						\
    cudaEventRecord(x ## _stop, 0);				\
    cudaEventSynchronize(x ## _stop);				\
    cudaEventElapsedTime(&dtime, x ## _start, x ## _stop);	\
    x ## time += dtime; }


void usage(){
  fprintf(stdout,
	  "process - process data from a PSRDADA ring buffer with input_key and\n"
	  "           write result to another PSRDADA ring buffer with output_key\n"
	  
	  "Usage: process [options]\n"
	  " -input_key/-i <key>       Hexadecimal shared memory key of input PSRDADA ring buffer [default: %x]\n"
	  " -beamform_key/-o <key>    Hexadecimal shared memory key of beamform output PSRDADA ring buffer [default: %x]\n"
	  " -gpu/-g <ID>              Run on ID GPU [default: 1]\n"
	  " -help/-h                  Show help\n",
	  DADA_DEFAULT_BLOCK_KEY,
	  DADA_DEFAULT_BLOCK_KEY+20
	  );
}

int main(int argc, char *argv[]){  
  struct option options[] = {
			     {"input_key",         1, 0, 'i'},
			     {"beamform_key",        1, 0, 'o'},
			     {"gpu",               1, 0, 'g'},
			     {"help",              0, 0, 'h'}, 
			     {0, 0, 0, 0}
  };

  key_t input_key = DADA_DEFAULT_BLOCK_KEY;
  key_t beamform_key = DADA_DEFAULT_BLOCK_KEY+20;
  int gpu = 0;
  int reset_bf = 1;

 // 读取解析各项输入参数 
  while (1) {
    unsigned ss;
    unsigned opt=getopt_long_only(argc, argv, "i:o:n:g:h", 
				  options, NULL);
    if (opt==EOF) break;
    
    switch (opt) {
      
    case 'i':
      key_t input_key_tmp;
      ss = sscanf(optarg, "%x", &input_key_tmp);
      if(ss != 1) {
        fprintf(stderr, "PROCESS_ERROR: Could not parse input key from %s, \n", optarg);
        fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
      else{
	      input_key = input_key_tmp;
      }
      break;

    case 'o':
      key_t output_key_tmp;
      ss = sscanf(optarg, "%x", &output_key_tmp);
      if(ss != 1) {
        fprintf(stderr, "PROCESS_ERROR: Could not parse output key from %s, \n", optarg);
        fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
      else{
        beamform_key = output_key_tmp;
      }
      break;

    case 'g':
      unsigned gpu_tmp;
      ss = sscanf(optarg, "%d", &gpu_tmp);
      if (ss!=1){
        fprintf(stderr, "PROCESS ERROR: Could not parse GPU id from %s, \n", optarg);
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
      else {
	      gpu = gpu_tmp;
      }
      break; 
    case 'h':
      usage();
      exit(EXIT_SUCCESS);
      
    case '?':
    default:
      break;
    }
  }

  fprintf(stdout, "DEBUG: gpu = %d\n", gpu);
  fprintf(stdout, "DEBUG: input_key = %x\n", input_key);
  fprintf(stdout, "DEBUG: beamform_key = %x\n", beamform_key);
  
  // Setup GPU with ID and print out its name
  cudaDeviceProp prop = {0};
  int gpu_get = gpuDeviceInit(gpu); // The required gpu might be different from what we get
  checkCudaErrors(cudaGetDeviceProperties(&prop, gpu_get));
  fprintf(stdout, "Asked for GPU %d, got GPU %d\n", gpu, gpu_get);
  fprintf(stdout, "GPU name is %s\n", prop.name);
  
  // Setup input dada ring buffer
  dada_hdu_t *input_hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(input_hdu, input_key);
  if(dada_hdu_connect(input_hdu) < 0){ 
    fprintf(stderr, "PROCESS_ERROR:\tCan not connect to input hdu with key %x"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    input_key, __FILE__, __LINE__);
    exit(EXIT_FAILURE);    
  }  
  ipcbuf_t *input_dblock = (ipcbuf_t *)(input_hdu->data_block);
  ipcbuf_t *input_hblock = (ipcbuf_t *)(input_hdu->header_block);
  
  if(dada_hdu_lock_read(input_hdu) < 0) {
    fprintf(stderr, "PROCESS_ERROR:\tError locking input HDU, \n"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have input HDU locked\n");
  fprintf(stdout, "PROCESS_INFO:\tWe have input HDU setup\n");

  // Setup beamform output ring buffer
  dada_hdu_t *beamform_hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(beamform_hdu, beamform_key);
  if(dada_hdu_connect(beamform_hdu) < 0){ 
    fprintf(stderr, "PROCESS_ERROR:\tCan not connect to beamform output hdu with key %x"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    beamform_key, __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  ipcbuf_t *beamform_output_dblock = (ipcbuf_t *)(beamform_hdu->data_block);
  ipcbuf_t *beamform_output_hblock = (ipcbuf_t *)(beamform_hdu->header_block);

  if(dada_hdu_lock_write(beamform_hdu) < 0) {
    fprintf(stderr, "PROCESS_ERROR:\tError locking beamform output HDU, \n"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have input HDU locked\n");
  fprintf(stdout, "PROCESS_INFO:\tWe have input HDU setup\n");

  // Now first read configuration from input ring buffer header & beamform.hh
  dada_header_t dada_header = {0};
  char *input_hbuf = ipcbuf_get_next_read(input_hblock,NULL);

  if (ascii_header_get(input_hbuf, "MJD_START", "%lf", &dada_header.mjd_start) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting MJD_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  
  double bf_tsamp = TSAMP*N_AVERAGE;

  
  fprintf(stdout, "HERE\n");
  fprintf(stdout, "DEBUG: gpu = %d\n", gpu);
  fprintf(stdout, "DEBUG: nelement = %d\n", N_ANTENNA);
  fprintf(stdout, "DEBUG: nbeam = %d\n", N_BEAM);
  fprintf(stdout, "DEBUG: nsamp_per_block = %d\n", N_TIMESTEP_PER_BLOCK);

  // 计算输入输出 ringbffer block的大小，并与dada header匹配
  int input_dbuf_size = BLOCK_SIZE;       //input ringbuffer
  int beamform_dbuf_size = INTE_BLOCK_SIZE; //beamform ringbuffer
  fprintf(stdout, "DEBUG: input_dbuf_size = %d\n", input_dbuf_size);
  fprintf(stdout, "DEBUG: beamform_dbuf_size = %d\n", beamform_dbuf_size);
  unsigned bytes_block_input  = ipcbuf_get_bufsz(input_dblock);
  unsigned bytes_block_beamform = ipcbuf_get_bufsz(beamform_output_dblock);

  fprintf(stdout, "PROCESS_INFO:\tinput buffer block size is %d bytes, output buffer block size is %d bytes\n",
	  bytes_block_input, bytes_block_beamform);
  
  if (bytes_block_input!=input_dbuf_size){
    fprintf(stderr, "PROCESS_ERROR:\tinput buffer block size mismatch, "
	    "%d vs %d "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    bytes_block_input, input_dbuf_size,
	    __FILE__, __LINE__);	
    
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have input buffer block size checked\n");

  if (bytes_block_beamform!=beamform_dbuf_size){
    fprintf(stderr, "PROCESS_ERROR:\tbeamform output buffer block size mismatch, "
	    "%d vs %d "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    bytes_block_beamform, beamform_dbuf_size,
	    __FILE__, __LINE__);	
    
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have beamform output buffer block size checked\n");


  // now we can setup new dada header buffer for output
  char *beamform_hbuf = ipcbuf_get_next_write (beamform_output_hblock);
  // setup beamform output ring buffer header
  if (ascii_header_set(beamform_hbuf, "MJD_START", "%lf", dada_header.mjd_start) < 0)  {
    fprintf(stderr, "BEAMFORM_ERROR: Error setting MJD_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(beamform_hbuf, "BF_NSAMP", "%d", N_TIMESTEP_PER_INIT_BLOCK) < 0)  {
    fprintf(stderr, "BEAMFORM_ERROR: Error setting BF_NSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(beamform_hbuf, "BF_NCHAN", "%d", N_CHAN) < 0)  {
    fprintf(stderr, "BEAMFORM_ERROR: Error setting NCHAN, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  } 

  if (ascii_header_set(beamform_hbuf, "BF_NBIT", "%d", BF_SAMPSZ) < 0)  {
    fprintf(stderr, "BEAMFORM_ERROR: Error setting NBIT, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(beamform_hbuf, "BF_TSAMP", "%f", bf_tsamp) < 0)  {
    fprintf(stderr, "BEAMFORM_ERROR: Error setting TSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  ipcbuf_mark_cleared(input_hblock);
  ipcbuf_mark_filled(beamform_output_hblock, DADA_DEFAULT_HEADER_SIZE);

  // Setup kernel dims for unpacking and integration
  dim3 unpack_dimgrid(N_ANTENNA,N_TIMESTEP_PER_BLOCK,1);
  dim3 unpack_dimblock(N_CHAN,1,1);
  dim3 inte_dimgrid(N_TIMESTEP_PER_INIT_BLOCK, N_CHAN, 1);
  dim3 inte_dimblock(N_BEAM,1,1);
  
  // setup beamformer
  int8_t *d_packed = NULL;
  cuComplex *d_A, *d_B, *d_C;
  int size_A = N_CHAN*N_TIMESTEP_PER_BLOCK*N_ANTENNA;
  int size_B = N_CHAN*N_ANTENNA*N_BEAM;
  int size_C = N_CHAN*N_TIMESTEP_PER_BLOCK*N_BEAM;
  float *h_B_real = (float *)malloc(sizeof(float)*size_B);
  float *h_B_imag = (float *)malloc(sizeof(float)*size_B);
  cuComplex *h_B = (cuComplex *)malloc(sizeof(cuComplex)*size_B);
  float *d_beamform_power;
  const cuComplex alpha(make_cuComplex(1,0));
  const cuComplex beta(make_cuComplex(0,0));

  for (int i=0; i<size_B; ++i){
        h_B_real[i] = 1;
        h_B_imag[i] = 0;
        h_B[i] = make_cuFloatComplex(*h_B_real,*h_B_imag);
  }
  checkCudaErrors(cudaMalloc(&d_packed, input_dbuf_size));
  checkCudaErrors(cudaMalloc(&d_A, size_A*sizeof(cuComplex)));     // element input
  checkCudaErrors(cudaMalloc(&d_B, size_B*sizeof(cuComplex)));     // beamform weights
  checkCudaErrors(cudaMalloc(&d_C, size_C*sizeof(cuComplex)));     // beamform output
  checkCudaErrors(cudaMalloc(&d_beamform_power,sizeof(float)*size_C));


  cublasHandle_t multi_plan;
  cublasCreate(&multi_plan);

  // Setup timer
  cudaEvent_t pipeline_start;
  cudaEvent_t pipeline_stop;
  float pipelinetime = 0;

  checkCudaErrors(cudaEventCreate(&pipeline_start));
  checkCudaErrors(cudaEventCreate(&pipeline_stop));

  cudaEvent_t memcpyh2d_start;
  cudaEvent_t memcpyh2d_stop;
  float memcpyh2dtime = 0;

  checkCudaErrors(cudaEventCreate(&memcpyh2d_start));
  checkCudaErrors(cudaEventCreate(&memcpyh2d_stop));
  
  cudaEvent_t memcpyd2h_start;
  cudaEvent_t memcpyd2h_stop;
  float memcpyd2htime = 0;

  checkCudaErrors(cudaEventCreate(&memcpyd2h_start));
  checkCudaErrors(cudaEventCreate(&memcpyd2h_stop));

  int nblock = 0;
  
  CUDA_STARTTIME(pipeline);  
  while(!ipcbuf_eod(input_dblock)){

    fprintf(stdout, "We are at %d block\n", nblock);
    // block memory copy,
    char *input_cbuf = ipcbuf_get_next_read(input_dblock, NULL);
    if(!input_cbuf){
      fprintf(stderr, "Could not get next read data block\n");
      exit(EXIT_FAILURE);
    }
    
    CUDA_STARTTIME(memcpyh2d);  
    checkCudaErrors(cudaMemcpy(d_packed, input_cbuf, bytes_block_input, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, size_B*sizeof(cuComplex), cudaMemcpyHostToDevice));
    CUDA_STOPTIME(memcpyh2d); 
    ipcbuf_mark_cleared(input_dblock);
    fprintf(stdout, "Memory copy from host to device of %d block done\n", nblock);

    krnl_unpack<<<unpack_dimgrid, unpack_dimblock>>>(d_packed, d_A, N_TIMESTEP_PER_BLOCK*N_ANTENNA);
    getLastCudaError("Kernel execution failed [ krnl_unpack ]");

    cublasCgemm3mStridedBatched(
                  multi_plan,
                  CUBLAS_OP_N,
                  CUBLAS_OP_N,
                  N_TIMESTEP_PER_BLOCK,
                  N_BEAM,
                  N_ANTENNA,
                  &alpha,
                  d_A,
                  N_TIMESTEP_PER_BLOCK,
                  N_TIMESTEP_PER_BLOCK*N_ANTENNA,
                  d_B,
                  N_ANTENNA,
                  N_ANTENNA*N_BEAM,
                  &beta,
                  d_C,
                  N_TIMESTEP_PER_BLOCK,
                  N_TIMESTEP_PER_BLOCK*N_BEAM,
                  N_CHAN);
    getLastCudaError("Kernel execution failed [ beamform ]");

    krnl_power_beamform<<<inte_dimgrid, inte_dimblock>>>(d_C,d_beamform_power,N_TIMESTEP_PER_INIT_BLOCK,N_AVERAGE);
    getLastCudaError("Kernel execution failed [ krnl_power_beamform ]");

    nblock++;
    //// 将输出的结果复制到输出ringbuffer
    char *output_bf = ipcbuf_get_next_write(beamform_output_dblock);
    if(!output_bf){
      fprintf(stderr, "Could not get next beamform write data block\n");
      exit(EXIT_FAILURE);
    }
    CUDA_STARTTIME(memcpyd2h);  
    checkCudaErrors(cudaMemcpy(output_bf, d_beamform_power, bytes_block_beamform, cudaMemcpyDeviceToHost));
    CUDA_STOPTIME(memcpyd2h);  
    ipcbuf_mark_filled(beamform_output_dblock, bytes_block_beamform);
 }
    
    CUDA_STOPTIME(pipeline);

    double available_time = TSAMP*N_PKT_PER_BLOCK/1.0E3;
    fprintf(stdout, "pipeline   %f milliseconds, pipline with memory transfer averaged with %d blocks\n", pipelinetime/(float)nblock, nblock);
    fprintf(stdout, "pipeline   %f milliseconds, memory transfer h2d averaged with %d blocks\n", memcpyh2dtime/(float)nblock, nblock);
    fprintf(stdout, "pipeline   %f milliseconds, beamform memory transfer d2h averaged with %d blocks\n", memcpyd2htime/(float)nblock, nblock);
    fprintf(stdout, "available  %f milliseconds, available time for pipeline to process a single block\n", available_time);
    
    dada_hdu_unlock_read(input_hdu);
    dada_hdu_unlock_write(beamform_hdu);
    checkCudaErrors(cublasDestroy(multi_plan));
    checkCudaErrors(cudaFree(d_packed));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_beamform_power));

    dada_hdu_destroy(input_hdu);
    dada_hdu_destroy(beamform_hdu);
    
    return EXIT_SUCCESS;
}    
