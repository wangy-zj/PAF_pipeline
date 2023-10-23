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
	  " -beamform_out/-o <key>    Hexadecimal shared memory key of beamform output PSRDADA ring buffer [default: %x]\n"
    " -zoom_out/-z <key>        Hexadecimal shared memory key of zoom FFT output PSRDADA ring buffer [default: %x]\n"
	  " -nthread/-n <N>           N thread per block for crossCorrPFT/crossCorr kernel [default: 64]\n"
	  " -gpu/-g <ID>              Run on ID GPU [default: 1]\n"
	  " -help/-h                  Show help\n",
	  DADA_DEFAULT_BLOCK_KEY,
	  DADA_DEFAULT_BLOCK_KEY+20,
    DADA_DEFAULT_BLOCK_KEY+40
	  );
}

__global__ void float_cucomplex(float *real, float *imag, cuComplex *output){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    output[index] = make_cuComplex(real[index],imag[index]);
}


int main(int argc, char *argv[]){  
  struct option options[] = {
			     {"input_key",         1, 0, 'i'},
			     {"beamform_out",        1, 0, 'o'},
           {"zoom_out",          1, 0, 'z'},
			     {"nthread",           1, 0, 'n'},
			     {"gpu",               1, 0, 'g'},
			     {"help",              0, 0, 'h'}, 
			     {0, 0, 0, 0}
  };

  key_t input_key = DADA_DEFAULT_BLOCK_KEY;
  key_t beamform_output_key = DADA_DEFAULT_BLOCK_KEY+20;
  key_t zoom_output_key = DADA_DEFAULT_BLOCK_KEY+40;
  int gpu = 0;
  int nthread = 128;
  int nthread_bf = 128;
  int reset_bf = 1;
  int reset_zoom = 1;

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
	beamform_output_key = output_key_tmp;
      }
      break;

    case 'z':
      key_t zoom_key_tmp;
      ss = sscanf(optarg, "%x", &zoom_key_tmp);
      if(ss != 1) {
	fprintf(stderr, "PROCESS_ERROR: Could not parse output key from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	usage();
	
	exit(EXIT_FAILURE);
      }
      else{
	zoom_output_key = zoom_key_tmp;
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
  fprintf(stdout, "DEBUG: beamform_output_key = %x\n", beamform_output_key);
  //fprintf(stdout, "DEBUG: zoom_output_key = %x\n", zoom_output_key);
  
  // Setup GPU with ID and print out its name
  cudaDeviceProp prop = {0};
  int gpu_get = gpuDeviceInit(gpu); // The required gpu might be different from what we get
  fprintf(stdout, "Asked for GPU %d, got GPU %d\n", gpu, gpu_get);
  checkCudaErrors(cudaGetDeviceProperties(&prop, gpu_get));
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
  dada_hdu_t *beamform_output_hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(beamform_output_hdu, beamform_output_key);
  if(dada_hdu_connect(beamform_output_hdu) < 0){ 
    fprintf(stderr, "PROCESS_ERROR:\tCan not connect to beamform output hdu with key %x"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    beamform_output_key, __FILE__, __LINE__);
    exit(EXIT_FAILURE);    
  }

  // setup zoomFFT output ring buffer
  dada_hdu_t *zoom_output_hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(zoom_output_hdu, zoom_output_key);
  if(dada_hdu_connect(zoom_output_hdu) < 0){ 
    fprintf(stderr, "PROCESS_ERROR:\tCan not connect to beamform output hdu with key %x"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    zoom_output_key, __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);    
  }

  ipcbuf_t *beamform_output_dblock = (ipcbuf_t *)(beamform_output_hdu->data_block);
  ipcbuf_t *beamform_output_hblock = (ipcbuf_t *)(beamform_output_hdu->header_block);

  if(dada_hdu_lock_write(beamform_output_hdu) < 0) {
    fprintf(stderr, "PROCESS_ERROR:\tError locking beamform output HDU, \n"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
      
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have beamform output HDU locked\n");
  fprintf(stdout, "PROCESS_INFO:\tWe have beamform output HDU setup\n");

  
  ipcbuf_t *zoom_output_dblock = (ipcbuf_t *)(zoom_output_hdu->data_block);
  ipcbuf_t *zoom_output_hblock = (ipcbuf_t *)(zoom_output_hdu->header_block);

  // make ourselves the write client
  if(dada_hdu_lock_write(zoom_output_hdu) < 0) {
    fprintf(stderr, "ZOOMFFT_ERROR:\tError locking zoomFFT output HDU, \n"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
      
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have zoomFFT output HDU locked\n");
  fprintf(stdout, "PROCESS_INFO:\tWe have zoomFFT output HDU setup\n");
  

  // Now first read configuration from input ring buffer header
  dada_header_t dada_header = {0};
  char *input_hbuf = ipcbuf_get_next_read(input_hblock, NULL);
  read_dada_header(input_hbuf, &dada_header);
  
  // 读取输入ring buffer header中的参数
  double mjd_start = dada_header.mjd_start;
  int npkt = dada_header.npkt;
  int Elements = dada_header.nelement;
  int Beams = dada_header.nbeam;
  int zoom_nchan = dada_header.zoom_nchan;
  int pkt_nsamp = dada_header.pkt_nsamp;
  int npol = dada_header.pkt_npol;
  double pkt_tsamp = dada_header.pkt_tsamp;
  int naverage_bf = dada_header.naverage_bf;
  int naverage_zoom = dada_header.naverage_zoom;
  int pkt_header_size = dada_header.pkt_header;
  int pkt_data_size = dada_header.pkt_data;
  //int pkt_size = pkt_header_size + pkt_data_size;
  
  // 根据输入ring buffer 参数计算输出ring buffer的参数
  int nsamp_packed = npkt*pkt_nsamp;    //每个block包含的samp数目

  // beamform ring buffer 参数
  int bf_nsamp = nsamp_packed/naverage_bf;
  int bf_nchan = 1;
  int bf_nbit = sizeof(float);
  double bf_tsamp = pkt_tsamp*naverage_bf;

  // zoom FFT ring buffer 参数
  int zoom_nsamp = nsamp_packed/zoom_nchan;
  int zoom_nbit = sizeof(float);
  double zoom_tsamp = zoom_nchan*pkt_tsamp;
  
  fprintf(stdout, "HERE\n");
  fprintf(stdout, "DEBUG: gpu = %d\n", gpu);
  fprintf(stdout, "DEBUG: npkt = %d\n", npkt);
  fprintf(stdout, "DEBUG: nelement = %d\n", Elements);
  fprintf(stdout, "DEBUG: nbeam = %d\n", Beams);
  fprintf(stdout, "DEBUG: nthread = %d\n", nthread);
  fprintf(stdout, "DEBUG: nsamp_packed = %d\n", nsamp_packed);

  // 计算输入输出 ringbffer block的大小，并与dada header匹配
  int input_dbuf_size = pkt_data_size*npkt*Elements*npol*sizeof(int8_t);       //input ringbuffer
  int beamform_dbuf_size = nsamp_packed*bf_nchan*Beams*bf_nbit/naverage_bf; //beamform ringbuffer
  int zoom_dbuf_size = Beams*zoom_nchan*zoom_nsamp*bf_nchan*zoom_nbit/naverage_zoom; //zoom fft ringbuffer
  fprintf(stdout, "DEBUG: input_dbuf_size = %d\n", input_dbuf_size);
  fprintf(stdout, "DEBUG: beamform_dbuf_size = %d\n", beamform_dbuf_size);
  fprintf(stdout, "DEBUG: zoom_dbuf_size = %d\n", zoom_dbuf_size);
  unsigned bytes_block_input  = ipcbuf_get_bufsz(input_dblock);
  unsigned bytes_block_beamform = ipcbuf_get_bufsz(beamform_output_dblock);
  unsigned bytes_block_zoom = ipcbuf_get_bufsz(zoom_output_dblock);

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

  
  if (bytes_block_zoom!=zoom_dbuf_size){
    fprintf(stderr, "PROCESS_ERROR:\tzoomFFT output buffer block size mismatch, "
	    "%d vs %d "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    bytes_block_zoom, zoom_dbuf_size,
	    __FILE__, __LINE__);	
    
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have zoomFFT output buffer block size checked\n");


  // now we can setup new dada header buffer for output
  char *beamform_hbuf = ipcbuf_get_next_write (beamform_output_hblock);
  memcpy(beamform_hbuf, input_hbuf, DADA_DEFAULT_HEADER_SIZE);
  // setup beamform output ring buffer header

  if (ascii_header_set(beamform_hbuf, "MJD_START", "%d", mjd_start) < 0)  {
    fprintf(stderr, "BEAMFORM_ERROR: Error setting MJD_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(beamform_hbuf, "BF_NSAMP", "%d", bf_nsamp) < 0)  {
    fprintf(stderr, "BEAMFORM_ERROR: Error setting BF_NSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(beamform_hbuf, "BF_NCHAN", "%d", bf_nchan) < 0)  {
    fprintf(stderr, "BEAMFORM_ERROR: Error setting NCHAN, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  } 

  if (ascii_header_set(beamform_hbuf, "BF_NBIT", "%d", bf_nbit) < 0)  {
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

  // zoom FFT ring buffer header
  char *zoom_hbuf = ipcbuf_get_next_write (beamform_output_hblock);
  memcpy(zoom_hbuf, input_hbuf, DADA_DEFAULT_HEADER_SIZE);
  // setup zoom FFT output ring buffer header
  if (ascii_header_set(zoom_hbuf, "MJD_START", "%d", mjd_start) < 0)  {
    fprintf(stderr, "ZOOMFFT_ERROR: Error setting MJD_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(zoom_hbuf, "ZOOM_NSAMP", "%d", zoom_nsamp) < 0)  {
    fprintf(stderr, "ZOOMFFT_ERROR: Error setting BF_NSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(zoom_hbuf, "ZOOM_NCHAN", "%d", zoom_nchan) < 0)  {
    fprintf(stderr, "ZOOMFFT_ERROR: Error setting NCHAN, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  } 

  if (ascii_header_set(zoom_hbuf, "ZOOM_NBIT", "%d", zoom_nbit) < 0)  {
    fprintf(stderr, "ZOOMFFT_ERROR: Error setting NBIT, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(zoom_hbuf, "ZOOM_TSAMP", "%f", zoom_tsamp) < 0)  {
    fprintf(stderr, "ZOOMFFT_ERROR: Error setting TSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  ipcbuf_mark_cleared(input_hblock);
  ipcbuf_mark_filled(beamform_output_hblock, DADA_DEFAULT_HEADER_SIZE);
  ipcbuf_mark_filled(zoom_output_hblock, DADA_DEFAULT_HEADER_SIZE);


  // Setup kernel dims
  dim3 grid_unpack(nsamp_packed/128+1);
  dim3 grid_taccumulate(nsamp_packed*Beams/nthread_bf+1);
  dim3 blck_pow(1, 1);
  dim3 grid_pow(1, 1);
	blck_pow.x = nthread;  
  grid_pow.x = (zoom_nchan - 1 + blck_pow.x) / blck_pow.x;
  

  // setup beamformer
  int8_t *d_packed = NULL;
  cuComplex *d_A, *d_B, *d_C, *d_zoom;
  int size_A = nsamp_packed*Elements;
  int size_B = Elements*Beams;
  int size_C = nsamp_packed*Beams;
  float *h_B_real = (float *)malloc(sizeof(float)*size_B);
  float *h_B_imag = (float *)malloc(sizeof(float)*size_B);
  cuComplex *h_B = (cuComplex *)malloc(sizeof(cuComplex)*size_B);
  float *d_beamform_power, *d_zoom_power;
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
  checkCudaErrors(cudaMalloc(&d_zoom, nsamp_packed * sizeof(cufftComplex)));
  checkCudaErrors(cudaMalloc(&d_zoom_power, nsamp_packed * sizeof(float)));

  //fprintf(stdout, "PROCESS_INFO:\t device input buffer size is %lud bytes\n", pkt_nsamp*sizeof(int8_t));
  
  cublasHandle_t multi_plan;
  cublasCreate(&multi_plan);

  // setup zoom FFT
  cufftHandle fftplan;
	const int rank = 1;   // 一维FFT
	int n[rank] = {zoom_nchan};
	int inembed[2] = {zoom_nchan, zoom_nsamp};
	int onembed[2] = {zoom_nchan, zoom_nsamp};
	int istride = 1;
	int idist = zoom_nchan;
	int ostride = 1;
	int odist = zoom_nchan;
	int batch = zoom_nsamp;

	cufftPlanMany(&fftplan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
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

  cudaEvent_t memcpyd2h_zoom_start;
  cudaEvent_t memcpyd2h_zoom_stop;
  float memcpyd2h_zoomtime = 0;

  checkCudaErrors(cudaEventCreate(&memcpyd2h_zoom_start));
  checkCudaErrors(cudaEventCreate(&memcpyd2h_zoom_stop));

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

    krnl_unpack<<<grid_unpack, 128>>>(d_packed, d_A, nsamp_packed*Elements, 8, 0);
    getLastCudaError("Kernel execution failed [ krnl_unpack ]");

    /*handle cublas 句柄
      transa 矩阵A是否转置行优先，需要时设置为CUBLAS_OP_T,不需要是设置为CUBLAS_OP_N
      transb 矩阵B是否转置行优先
      m 矩阵A和C的行数
      n 矩阵B和C的列数
      k 矩阵A的列数和B的行数
      alpha AB相乘的系数

      beta 与C相加的系数
      A，B，C 输入矩阵A，B，C；矩阵C为列优先
      lda 若不转置，为A的行数(m)，否则为A的列数(k)
      ldb 若不转置，为B的行数(k)，否则为B的列数(n)
      ldc C的行数(m) */
    cublasCgemm3m(
                  multi_plan,
                  CUBLAS_OP_N,
                  CUBLAS_OP_N,
                  Beams,
                  nsamp_packed,
                  Elements,
                  &alpha,
                  d_B,
                  Beams,
                  d_A,
                  Elements,
                  &beta,
                  d_C,
                  Beams);
    getLastCudaError("Kernel execution failed [ beamform ]");

    krnl_power_beamform<<<grid_taccumulate, nthread_bf>>>(d_C,d_beamform_power,nsamp_packed/naverage_bf,naverage_bf,reset_bf);
    getLastCudaError("Kernel execution failed [ krnl_power_beamform ]");

    
    //补充zoom fft代码，输入为beamform的输出d_C，zoom_nchan为ZOOM FFT的通道数，输出为d_zoom
    cufftExecC2C(fftplan, d_C, d_zoom, CUFFT_FORWARD);
    krnl_power_zoomfft<<<grid_pow, blck_pow>>>(d_zoom,d_zoom_power, zoom_nsamp, zoom_nchan,reset_zoom);
    getLastCudaError("Kernel execution fialed [ krnl_power_zoom ]");
    
    nblock++;
    //// 将输出的结果复制到输出ringbuffer
    if(nblock % naverage_bf ==0){
    reset_bf = 1;
    char *output_bf = ipcbuf_get_next_write(beamform_output_dblock);
    if(!output_bf){
      fprintf(stderr, "Could not get next beamform write data block\n");
      exit(EXIT_FAILURE);
    }
    CUDA_STARTTIME(memcpyd2h);  
    checkCudaErrors(cudaMemcpy(output_bf, d_beamform_power, bytes_block_beamform, cudaMemcpyDeviceToHost));
    CUDA_STOPTIME(memcpyd2h);  
    ipcbuf_mark_filled(beamform_output_dblock, bytes_block_beamform);
    fprintf(stdout, "we copy beamform data out\n");
  }else{
    reset_bf = 0;
  }

    if(nblock % naverage_zoom == 0){
    reset_zoom = 1;
    char *output_zoom = ipcbuf_get_next_write(zoom_output_dblock);
    if(!output_zoom){
      fprintf(stderr, "Could not get next zoom write data block\n");
      exit(EXIT_FAILURE);
    }
    CUDA_STARTTIME(memcpyd2h_zoom);  
    checkCudaErrors(cudaMemcpy(output_zoom, d_zoom_power, bytes_block_zoom, cudaMemcpyDeviceToHost));
    CUDA_STOPTIME(memcpyd2h_zoom);  
    ipcbuf_mark_filled(zoom_output_dblock, bytes_block_zoom);
    fprintf(stdout, "we copy zoom data out\n");
    }else{
      reset_zoom = 0;
    }
  }
    
    CUDA_STOPTIME(pipeline);

    double available_time = pkt_tsamp*npkt/1.0E3;
    fprintf(stdout, "pipeline   %f milliseconds, pipline with memory transfer averaged with %d blocks\n", pipelinetime/(float)nblock, nblock);
    fprintf(stdout, "pipeline   %f milliseconds, memory transfer h2d averaged with %d blocks\n", memcpyh2dtime/(float)nblock, nblock);
    fprintf(stdout, "pipeline   %f milliseconds, beamform memory transfer d2h averaged with %d blocks\n", memcpyd2htime/(float)nblock, nblock);
    fprintf(stdout, "pipeline   %f milliseconds, zoom memory transfer d2h averaged with %d blocks\n", memcpyd2htime/(float)nblock, nblock);
    fprintf(stdout, "available  %f milliseconds, available time for pipeline to process a single block\n", available_time);
    
    dada_hdu_unlock_read(input_hdu);
    dada_hdu_unlock_write(beamform_output_hdu);
    dada_hdu_unlock_write(zoom_output_hdu);
    checkCudaErrors(cublasDestroy(multi_plan));
    checkCudaErrors(cufftDestroy(fftplan)); 
    checkCudaErrors(cudaFree(d_packed));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_beamform_power));
    checkCudaErrors(cudaFree(d_zoom));
    checkCudaErrors(cudaFree(d_zoom_power));

    dada_hdu_destroy(input_hdu);
    dada_hdu_destroy(beamform_output_hdu);
    dada_hdu_destroy(zoom_output_hdu);
    
    return EXIT_SUCCESS;
}    
