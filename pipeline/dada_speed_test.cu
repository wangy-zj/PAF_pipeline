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
	  "process - process data from a PSRDADA ring buffer with input_key and\n"
	  "           write result to another PSRDADA ring buffer with output_key\n"
	  "Usage: process [options]\n"
	  " -input_key/-i <key>       Hexadecimal shared memory key of input PSRDADA ring buffer [default: %x]\n"
	  " -output_key/-o <key>      Hexadecimal shared memory key of output PSRDADA ring buffer [default: %x]\n"
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
			     {"gpu",               1, 0, 'g'},
			     {"help",              0, 0, 'h'}, 
			     {0, 0, 0, 0}
  };

  int input_key = DADA_DEFAULT_BLOCK_KEY;
  int output_key = DADA_DEFAULT_BLOCK_KEY+20;
  int gpu = 0;

  while (1) {
    unsigned ss;
    unsigned opt=getopt_long_only(argc, argv, "i:o:g:h", 
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
    case 'h':
      usage();
      exit(EXIT_SUCCESS);
      
    case '?':
    default:
      break;
    }
  }

// 绑定输入ringbuffer
  dada_hdu_t *input_hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(input_hdu, input_key);
  if(dada_hdu_connect(input_hdu) < 0){ 
    fprintf(stderr, "PROCESS_ERROR:\tCan not connect to input hdu with key %x\n"
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


// 绑定输出ringbuffer
  dada_hdu_t *output_hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(output_hdu, output_key);
  if(dada_hdu_connect(output_hdu) < 0){ 
    fprintf(stderr, "PROCESS_ERROR:\tCan not connect to output hdu with key %x\n"
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
  fprintf(stdout, "PROCESS_INFO:\tWe have output HDU setup\n");

// copy ringbuffer header
  //memcpy(output_hblock, input_hblock, sizeof(input_hdu->header_block));

  // set up CUDA
  while(!ipcbuf_eod(input_dblock)){

    //fprintf(stdout, "We are at %d block\n", nblock);
    // block memory copy,
    char *input_cbuf = ipcbuf_get_next_read(input_dblock, NULL);
    if(!input_cbuf){
      fprintf(stderr, "Could not get next read data block\n");
      exit(EXIT_FAILURE);
    }
    char *output_cbuf = ipcbuf_get_next_write(output_dblock);
    if(!output_cbuf){
      fprintf(stderr, "Could not get next write data block\n");
      exit(EXIT_FAILURE);
    }
    unsigned bytes_block_input  = ipcbuf_get_bufsz(input_dblock);
    checkCudaErrors(cudaMemcpy(output_cbuf, input_cbuf, bytes_block_input,cudaMemcpyHostToDevice));
    ipcbuf_mark_cleared(input_dblock);
    ipcbuf_mark_filled(output_dblock,bytes_block_input);
  }
  dada_hdu_unlock_read(input_hdu);
  dada_hdu_unlock_write(output_hdu);
  dada_hdu_destroy(input_hdu);
  dada_hdu_destroy(output_hdu);
    
  return EXIT_SUCCESS;
}

