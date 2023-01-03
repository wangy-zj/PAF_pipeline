#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
  This is the main function to test krnl_unpack_bigcat kernel.
  the kernel does not produce dig value?
*/

#include "dada_header.h"
#include "dada_def.h"
#include "futils.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {

  char dada_header_buffer[DADA_DEFAULT_HEADER_SIZE];
  char dada_header_filename[DADA_STRLEN] = {"0"};

  strcpy(dada_header_filename, "../dada.header");
  
  if(fileread(dada_header_filename, dada_header_buffer, DADA_DEFAULT_HEADER_SIZE) < 0){
    fprintf(stderr, "TEST_GENERATED_CODE_ERROR:\tError reading DADA header file, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
    
    // Have to error out if we could not read header template file
    exit(EXIT_FAILURE);
  }

  dada_header_t dada_header = {0};

  read_dada_header(dada_header_buffer, &dada_header);

  fprintf(stdout, "TSAMP: %f\n",     dada_header.tsamp);
  fprintf(stdout, "BW: %f\n",        dada_header.bw);
  fprintf(stdout, "MJD_START: %f\n", dada_header.mjd_start);
  fprintf(stdout, "UTC_START: %s\n", dada_header.utc_start);
  fprintf(stdout, "NCHAN: %d\n\n",   dada_header.nchan);
  
  dada_header.tsamp = 10.1;
  dada_header.bw = 101.0;
  dada_header.mjd_start = 10000;
  strcpy(dada_header.utc_start, "2022-02-02");
  dada_header.nchan =1024;

  
  write_dada_header(dada_header, dada_header_buffer);
  read_dada_header(dada_header_buffer, &dada_header);
  
  fprintf(stdout, "TSAMP: %f\n",     dada_header.tsamp);
  fprintf(stdout, "BW: %f\n",        dada_header.bw);
  fprintf(stdout, "MJD_START: %f\n", dada_header.mjd_start);
  fprintf(stdout, "UTC_START: %s\n", dada_header.utc_start);
  fprintf(stdout, "NCHAN: %d\n\n",   dada_header.nchan);

  // write data to a file
  write_dada_header_to_file(dada_header, "write_header.txt");
  
  read_dada_header_from_file("write_header.txt", &dada_header);

  fprintf(stdout, "HERE\n");
  
  fprintf(stdout, "TSAMP: %f\n",     dada_header.tsamp);
  fprintf(stdout, "BW: %f\n",        dada_header.bw);
  fprintf(stdout, "MJD_START: %f\n", dada_header.mjd_start);
  fprintf(stdout, "UTC_START: %s\n", dada_header.utc_start);
  fprintf(stdout, "NCHAN: %d\n\n",   dada_header.nchan);

  
  return EXIT_SUCCESS;
}
