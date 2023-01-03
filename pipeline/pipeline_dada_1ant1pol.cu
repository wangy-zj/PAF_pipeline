#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
  This is the main function to check the performance of process
*/

#include <getopt.h>


#include "../include/cuda/cuda_utilities.h"
#include "../include/test.h"
#include "../include/krnl.h"
#include "../include/dada_header.h"
#include "../include/dada_util.h"

// better to put this here
#include "dada_cuda.h"

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
	  " -output_key/-o <key>      Hexadecimal shared memory key of output PSRDADA ring buffer [default: %x]\n"
	  " -nthread/-n <N>           N thread per block for crossCorrPFT/crossCorr kernel [default: 64]\n"
	  " -gpu/-g <ID>              Run on ID GPU [default: 1]\n"
	  " -help/-h                  Show help\n",
	  DADA_DEFAULT_BLOCK_KEY,
	  DADA_DEFAULT_BLOCK_KEY+20
	  );
}

int main(int argc, char *argv[]){  
  struct option options[] = {
			     {"input_key",         1, 0, 'i'},
			     {"output_key",        1, 0, 'o'},
			     {"nthread",           1, 0, 'n'},
			     {"gpu",               1, 0, 'g'},
			     {"help",              0, 0, 'h'}, 
			     {0, 0, 0, 0}
  };

  key_t input_key = DADA_DEFAULT_BLOCK_KEY;
  key_t output_key = DADA_DEFAULT_BLOCK_KEY+20;
  int gpu = 0;
  int nthread = 128;
  int reset = 1;

 // 读取各项输入参数 
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
	usage();
	
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
	usage();
	
	exit(EXIT_FAILURE);
      }
      else{
	output_key = output_key_tmp;
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

    case 'n':
      unsigned nthread_tmp;
      ss = sscanf(optarg, "%d", &nthread_tmp);
      if (ss!=1){
        fprintf(stderr, "PROCESS ERROR: Could not parse nthread from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);

	exit(EXIT_FAILURE);
      }
      else {
	nthread = nthread_tmp;
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
  fprintf(stdout, "DEBUG: nthread = %d\n", nthread);

  fprintf(stdout, "DEBUG: input_key = %x\n", input_key);
  fprintf(stdout, "DEBUG: output_key = %x\n", output_key);
  
  // Setup GPU with ID and print out its name
  cudaDeviceProp prop = {0};
  int gpu_get = gpuDeviceInit(gpu); // The required gpu might be different from what we get
  fprintf(stdout, "Asked for GPU %d, got GPU %d\n", gpu, gpu_get);
  checkCudaErrors(cudaGetDeviceProperties(&prop, gpu_get));
  fprintf(stdout, "GPU name is %s\n", prop.name);
  
  // Get hold of dada ring buffers
    // Setup input dada ring buffer
  // lock hdu will be in loop
  dada_hdu_t *input_hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(input_hdu, input_key);
  if(dada_hdu_connect(input_hdu) < 0){ 
    fprintf(stderr, "PROCESS_ERROR:\tCan not connect to input hdu with key %x"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    input_key, __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);
  }  
  // registers the existing host memory range for use by CUDA
  // dada_cuda_dbregister(input_hdu);   
  ipcbuf_t *input_dblock = (ipcbuf_t *)(input_hdu->data_block); // ->取成员运算符
  ipcbuf_t *input_hblock = (ipcbuf_t *)(input_hdu->header_block);
  
  if(dada_hdu_lock_read(input_hdu) < 0) {
    fprintf(stderr, "PROCESS_ERROR:\tError locking input HDU, \n"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
      
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have input HDU locked\n");
  fprintf(stdout, "PROCESS_INFO:\tWe have input HDU setup\n");
  
  // Setup output ring buffer
  dada_hdu_t *output_hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(output_hdu, output_key);
  if(dada_hdu_connect(output_hdu) < 0){ 
    fprintf(stderr, "PROCESS_ERROR:\tCan not connect to output hdu with key %x"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    output_key, __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);    
  } 	   
  ipcbuf_t *output_dblock = (ipcbuf_t *)(output_hdu->data_block);
  ipcbuf_t *output_hblock = (ipcbuf_t *)(output_hdu->header_block);
  
  // make ourselves the write client
  if(dada_hdu_lock_write(output_hdu) < 0) {
    fprintf(stderr, "PROCESS_ERROR:\tError locking output HDU, \n"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
      
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have output HDU locked\n");
  fprintf(stdout, "PROCESS_INFO:\tWe have output HDU setup\n");
  
  // Now first read configuration from input ring buffer header
  dada_header_t dada_header = {0};
  char *input_hbuf = ipcbuf_get_next_read(input_hblock, NULL);
  read_dada_header(input_hbuf, &dada_header);
  
  // parse values from header buffer 从header buffer 解析值
  int npkt = dada_header.npkt;
  int pkt_nsamp = dada_header.pkt_nsamp;
  int nchan_fine = dada_header.nchan_fine;
  double pkt_tsamp = dada_header.pkt_tsamp;
  int naverage = dada_header.naverage;
  
  int nsamp_packed = npkt*pkt_nsamp;
  int nfft_point = (nchan_fine-1)*2;
  
  int nfft = nsamp_packed/nfft_point;
  int nsamp_fft = nfft*nchan_fine;
  fprintf(stdout, "HERE\n");
  
  fprintf(stdout, "DEBUG: gpu = %d\n", gpu);
  fprintf(stdout, "DEBUG: npkt = %d\n", npkt);
  fprintf(stdout, "DEBUG: nfft = %d\n", nfft);
  fprintf(stdout, "DEBUG: nchan_fine = %d\n", nchan_fine);
  fprintf(stdout, "DEBUG: nthread = %d\n", nthread);
  fprintf(stdout, "DEBUG: nsamp_packed = %d\n", nsamp_packed);
  fprintf(stdout, "DEBUG: nsamp_fft = %d\n", nsamp_fft);

  // Need to check it against expected value here
  // 设置输入输出的buffer大小
  int input_dbuf_size = nsamp_packed*sizeof(int8_t);
  int output_dbuf_size = nchan_fine*sizeof(float);
  
  unsigned bytes_block_input  = ipcbuf_get_bufsz(input_dblock);
  unsigned bytes_block_output = ipcbuf_get_bufsz(output_dblock);  

  fprintf(stdout, "PROCESS_INFO:\tinput buffer block size is %d bytes, output buffer block size is %d bytes\n",
	  bytes_block_input, bytes_block_output);
  
  if (bytes_block_input!=input_dbuf_size){
    fprintf(stderr, "PROCESS_ERROR:\tinput buffer block size mismatch, "
	    "%d vs %d "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    bytes_block_input, input_dbuf_size,
	    __FILE__, __LINE__);	
    
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have input buffer block size checked\n");

  if (bytes_block_output!=output_dbuf_size){
    fprintf(stderr, "PROCESS_ERROR:\toutput buffer block size mismatch, "
	    "%d vs %d "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    bytes_block_output, output_dbuf_size,
	    __FILE__, __LINE__);	
    
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have output buffer block size checked\n");

  // now we can setup new dada header buffer for output
  char *output_hbuf = ipcbuf_get_next_write(output_hblock);
  double tsamp_output = npkt*pkt_tsamp*naverage;
  memcpy(output_hbuf, input_hbuf, DADA_DEFAULT_HEADER_SIZE);
  // We update tsamp
  if (ascii_header_set(output_hbuf, "TSAMP", "%f", tsamp_output) < 0)  {
    fprintf(stderr, "PROCESS_ERROR:\tError setting TSAMP, "
	    "which happens at \"%s\", line [%d].\n",
	    __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  // We update file size
  // a single buffer block per file
  // BYTES_PER_SECOND also need update, but ignore it for now
  if (ascii_header_set(output_hbuf, "FILE_SIZE", "%d", bytes_block_output) < 0)  {
    fprintf(stderr, "PROCESS_ERROR:\tError setting FILE_SIZE, "
	    "which happens at \"%s\", line [%d].\n",
	    __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  ipcbuf_mark_cleared(input_hblock);
  ipcbuf_mark_filled(output_hblock, DADA_DEFAULT_HEADER_SIZE);

  // Setup kernel dims
  dim3 grid_unpack(nsamp_packed/nthread+1);
  dim3 grid_taccumulate(nchan_fine/nthread+1);
  
  // Setup cuda buffers
  int8_t *d_packed = NULL;
  cuComplex *d_ffted = NULL;
  float *d_unpacked = NULL;
  float *d_result = NULL;
  
  checkCudaErrors(cudaMalloc(&d_packed, input_dbuf_size));
  checkCudaErrors(cudaMalloc(&d_unpacked, nsamp_packed*sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_ffted, nsamp_fft*sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_result, output_dbuf_size));

  fprintf(stdout, "PROCESS_INFO:\t device input buffer size is %lud bytes\n", pkt_nsamp*sizeof(int8_t));
  
  // setup FFT
  cufftHandle fft_plan;
  checkCudaErrors(cufftPlan1d(&fft_plan, nfft_point, CUFFT_R2C, nfft));
  print_cuda_memory_info();

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
    
    // memory copy 
    CUDA_STARTTIME(memcpyh2d);  
    checkCudaErrors(cudaMemcpy(d_packed, input_cbuf, bytes_block_input, cudaMemcpyHostToDevice));
    CUDA_STOPTIME(memcpyh2d);  
    ipcbuf_mark_cleared(input_dblock);
    fprintf(stdout, "Memory copy from host to device of %d block done\n", nblock);
    
    // unpack the data block
    krnl_unpack_1ant1pol<<<grid_unpack, nthread>>>(d_packed, d_unpacked, nsamp_packed);
    getLastCudaError("Kernel execution failed [ krnl_unpack_1ant1pol ]");
    
    // 一维FFT
    checkCudaErrors(cufftExecR2C(fft_plan, d_unpacked, d_ffted));
    
    // 能量积分
    krnl_power_taccumulate_1ant1pol<<<grid_taccumulate, nthread>>>(d_ffted, d_result, nfft, nchan_fine, reset);
    getLastCudaError("Kernel execution failed [ krnl_power_taccumulate_1ant1pol ]");
    
    nblock++;
    if(nblock%naverage == 0){
      reset = 1;
    
      //// we copy data to ring buffer only when we get naverage blocks done
      //// block memory copy
      //// We will not get any output if the runtime is short than integration time
      char *output_cbuf = ipcbuf_get_next_write(output_dblock);
      if(!output_cbuf){
      	fprintf(stderr, "Could not get next write data block\n");
      	exit(EXIT_FAILURE);
      }
      // 将运算结果cpoy到output buffer
      CUDA_STARTTIME(memcpyd2h);  
      checkCudaErrors(cudaMemcpy(output_cbuf, d_result, bytes_block_output, cudaMemcpyDeviceToHost));
      CUDA_STOPTIME(memcpyd2h);  
      ipcbuf_mark_filled(output_dblock, bytes_block_output);

      fprintf(stdout, "we copy data out\n");
    }
    else{
      reset = 0;
    }
  }
  
  CUDA_STOPTIME(pipeline);

  double available_time = pkt_tsamp*npkt/1.0E3;
  fprintf(stdout, "pipeline   %f milliseconds, pipline with memory transfer averaged with %d blocks\n", pipelinetime/(float)nblock, nblock);
  fprintf(stdout, "pipeline   %f milliseconds, memory transfer h2d averaged with %d blocks\n", memcpyh2dtime/(float)nblock, nblock);
  fprintf(stdout, "pipeline   %f milliseconds, memory transfer d2h averaged with %d blocks\n", memcpyd2htime/(float)nblock, nblock);
  fprintf(stdout, "available  %f milliseconds, available time for pipeline to process a single block\n", available_time);
  
  dada_hdu_unlock_read(input_hdu);
  dada_hdu_unlock_write(output_hdu);
  checkCudaErrors(cufftDestroy(fft_plan));
  
  dada_hdu_destroy(input_hdu);
  dada_hdu_destroy(output_hdu);
  
  return EXIT_SUCCESS;
}    
