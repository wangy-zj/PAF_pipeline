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


void usage(){
  fprintf(stdout,
	  "process - mark the input_key ringbuffer as clear\n"
	  "Usage: process [options]\n"
	  " -input_key/-i <key>       Hexadecimal shared memory key of input PSRDADA ring buffer [default: %x]\n"
	  " -help/-h                  Show help\n",
	  DADA_DEFAULT_BLOCK_KEY
	  );
}

int main(int argc, char *argv[]){  
  struct option options[] = {
			     {"input_key",         1, 0, 'i'},
			     {"help",              0, 0, 'h'}, 
			     {0, 0, 0, 0}
  };

  int input_key = DADA_DEFAULT_BLOCK_KEY;

  while (1) {
    unsigned ss;
    unsigned opt=getopt_long_only(argc, argv, "i:g:h", 
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

    case 'h':
      usage();
      exit(EXIT_SUCCESS);
      
    case '?':
    default:
      break;
    }
  }

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
  fprintf(stdout, "PROCESS_INFO:\tWe have cleared HDU locked\n");

  while(!ipcbuf_eod(input_dblock)){

    //fprintf(stdout, "We are at %d block\n", nblock);
    // block memory copy,
    char *input_cbuf = ipcbuf_get_next_read(input_dblock, NULL);
    if(!input_cbuf){
      fprintf(stderr, "Could not get next read data block\n");
      exit(EXIT_FAILURE);
    }
    ipcbuf_mark_cleared(input_dblock);
  }
  dada_hdu_unlock_read(input_hdu);
  dada_hdu_destroy(input_hdu);

  return EXIT_SUCCESS;
}