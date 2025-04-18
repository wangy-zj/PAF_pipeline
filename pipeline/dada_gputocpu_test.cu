#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

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

void usage(){
  fprintf(stdout,
	  "process - process data from a PSRDADA ring buffer with gpu_key and\n"
	  "           write result to another PSRDADA ring buffer with output_key\n"
	  "Usage: process [options]\n"
	  " -gpu_key/-g <key>       Hexadecimal shared memory key of GPU PSRDADA ring buffer [default: %x]\n"
	  " -output_key/-o <key>      Hexadecimal shared memory key of output PSRDADA ring buffer [default: %x]\n"
	  " -help/-h                  Show help\n",
	  DADA_DEFAULT_BLOCK_KEY,
	  DADA_DEFAULT_BLOCK_KEY+10
	  );
}

int main(int argc, char *argv[]){  
  struct option options[] = {
			     {"gpu_key",         1, 0, 'g'},
			     {"output_key",      1, 0, 'o'},
			     {"help",            0, 0, 'h'}, 
			     {0, 0, 0, 0}
  };

  int gpu_key = DADA_DEFAULT_BLOCK_KEY;
  int output_key = DADA_DEFAULT_BLOCK_KEY+10;

  while (1) {
    unsigned ss;
    unsigned opt=getopt_long_only(argc, argv, "g:o:h", 
				  options, NULL);
    if (opt==EOF) break;
    
    switch (opt) {
      
    case 'g':
      key_t gpu_key_tmp;
      ss = sscanf(optarg, "%x", &gpu_key_tmp);
      if(ss != 1) {
        fprintf(stderr, "PROCESS_ERROR: Could not parse input key from %s, \n", optarg);
        fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
      else{
	      gpu_key = gpu_key_tmp;
      }
      break;

    case 'o':
      key_t output_key_tmp;
      ss = sscanf(optarg, "%x", &output_key_tmp);
      if(ss != 1) {
        fprintf(stderr, "PROCESS_ERROR: CoHDUt \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
      else{
        output_key = output_key_tmp;
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

// 绑定GPU ringbuffer
  dada_hdu_t *gpu_hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(gpu_hdu, gpu_key);
  if(dada_hdu_connect(gpu_hdu) < 0){ 
    fprintf(stderr, "PROCESS_ERROR:\tCan not connect to output hdu with key %x\n"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    gpu_key, __FILE__, __LINE__);
    exit(EXIT_FAILURE);    
  }  
  ipcbuf_t *gpu_dblock = (ipcbuf_t *)(gpu_hdu->data_block);
  ipcbuf_t *gpu_hblock = (ipcbuf_t *)(gpu_hdu->header_block);
  
  if(dada_hdu_lock_read(gpu_hdu) < 0) {
    fprintf(stderr, "PROCESS_ERROR:\tError locking GPU HDU write, \n"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have GPU HDU locked\n");

  // 绑定输出ringbuffer
  dada_hdu_t *output_hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(output_hdu, output_key);
  if(dada_hdu_connect(output_hdu) < 0){ 
    fprintf(stderr, "PROCESS_ERROR:\tCan not connect to input hdu with key %x\n"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    output_key, __FILE__, __LINE__);
    exit(EXIT_FAILURE);    
  }  
  ipcbuf_t *output_dblock = (ipcbuf_t *)(output_hdu->data_block);
  ipcbuf_t *output_hblock = (ipcbuf_t *)(output_hdu->header_block);
  
  if(dada_hdu_lock_write(output_hdu) < 0) {
    fprintf(stderr, "PROCESS_ERROR:\tError locking output HDU, \n"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have output HDU locked\n");

// copy ringbuffer header
  char *gpu_hbuf = ipcbuf_get_next_read(gpu_hblock,NULL);
  if (!gpu_hbuf){
    fprintf(stderr, "Could not get next read header block\n");
    exit(EXIT_FAILURE);
  }

  char *output_hbuf = ipcbuf_get_next_write(output_hblock);
  if (!output_hbuf){
    fprintf(stderr, "Could not get next GPU write header block\n");
    exit(EXIT_FAILURE);
  }
  memcpy(output_hbuf, gpu_hbuf, DADA_DEFAULT_HEADER_SIZE);
  
  ipcbuf_mark_cleared(gpu_hblock);
  ipcbuf_mark_filled(output_hblock,DADA_DEFAULT_HEADER_SIZE);

  int block_num = 1;
  // set up CUDA
  while(!ipcbuf_eod(gpu_dblock)){

    //fprintf(stdout, "We are at %d block\n", nblock);
    // block memory copy,
    char *gpu_cbuf = ipcbuf_get_next_read(gpu_dblock, NULL);
    if(!gpu_cbuf){
      fprintf(stderr, "Could not get next read data block\n");
      exit(EXIT_FAILURE);
    }
    char *output_ibuf = ipcbuf_get_next_write(output_dblock);
    if(!output_ibuf){
      fprintf(stderr, "Could not get next gpu write data block\n");
      exit(EXIT_FAILURE);
    }
    
    //add beamform here

    //add Zoom FFT here

    //add intergration here

    //copy data to CPU
    unsigned bytes_block_input  = ipcbuf_get_bufsz(gpu_dblock);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    checkCudaErrors(cudaMemcpyAsync(output_ibuf, gpu_cbuf, bytes_block_input,cudaMemcpyDeviceToHost,stream));
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    ipcbuf_mark_cleared(gpu_dblock);
    ipcbuf_mark_filled(output_dblock,bytes_block_input);


    if(block_num%100==0){
      fprintf(stdout, "We have copied %d blocks from GPU to CPU\n", block_num);
    }
    block_num++;
  }
  dada_hdu_unlock_read(gpu_hdu);
  dada_hdu_unlock_write(output_hdu);
  dada_hdu_destroy(output_hdu);
  dada_hdu_destroy(gpu_hdu);
    
  return EXIT_SUCCESS;
}

