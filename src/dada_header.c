#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "futils.h"
#include "dada_def.h"
#include "ascii_header.h"

#include "../include/dada_header.h"

#include <stdlib.h>
#include <stdio.h>

#include <string.h>

int read_dada_header_from_file(const char *dada_header_file_name, dada_header_t *dada_header){

  char *dada_header_buffer = (char *)malloc(DADA_DEFAULT_HEADER_SIZE);
  memset(dada_header_buffer, 0, DADA_DEFAULT_HEADER_SIZE);

  fileread(dada_header_file_name, dada_header_buffer, DADA_DEFAULT_HEADER_SIZE);
  read_dada_header(dada_header_buffer, dada_header);

  free(dada_header_buffer);

  return EXIT_SUCCESS;
}

int read_dada_header(const char *dada_header_buffer, dada_header_t *dada_header){


  if (ascii_header_get(dada_header_buffer, "MJD_START", "%lf", &dada_header->mjd_start) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting MJD_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "NPKT", "%d", &dada_header->npkt) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting NPKT, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "NELEMENT", "%d", &dada_header->nelement) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting RECEIVER, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "NBEAM", "%d", &dada_header->nbeam) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting BEAM, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "ZOOM_NCHAN", "%d", &dada_header->zoom_nchan) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting ZOOM_NCHAN, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "PKT_HEADER", "%d", &dada_header->pkt_header) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting PKT_HEADER, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "PKT_DATA", "%d", &dada_header->pkt_data) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting PKT_DATA, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "PKT_NSAMP", "%d", &dada_header->pkt_nsamp) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting PKT_NSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "PKT_NCHAN", "%d", &dada_header->pkt_nchan) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting PKT_NCHAN, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "PKT_NPOL", "%d", &dada_header->pkt_npol) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting NPOL, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "PKT_NBIT", "%d", &dada_header->pkt_nbit) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting NBIT, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "PKT_TSAMP", "%lf", &dada_header->pkt_tsamp) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting PKT_TSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "NAVERAGE_BF", "%d", &dada_header->naverage_bf) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting NAVERAGE_BF, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "NAVERAGE_ZOOM", "%d", &dada_header->naverage_zoom) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting NAVERAGE_ZOOM, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}

int write_dada_header_to_file(const dada_header_t dada_header, const char *dada_header_file_name){

  FILE *fp = fopen(dada_header_file_name, "w");
  char *dada_header_buffer = (char *)malloc(DADA_DEFAULT_HEADER_SIZE);
  memset(dada_header_buffer, 0, DADA_DEFAULT_HEADER_SIZE);

  sprintf(dada_header_buffer, "HDR_VERSION  1.0\nHDR_SIZE     4096\n");
  write_dada_header(dada_header, dada_header_buffer);
  fprintf(fp, "%s\n", dada_header_buffer);

  free(dada_header_buffer);
  fclose(fp);

  return EXIT_SUCCESS;
}

int write_dada_header(const dada_header_t dada_header, char *dada_header_buffer){

  if (ascii_header_set(dada_header_buffer, "MJD_START", "%f", dada_header.mjd_start) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting MJD_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "NPKT", "%d", dada_header.npkt) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting NPKT, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "NELEMENT", "%s", dada_header.nelement) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting NELEMENT, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "NBEAM", "%d", dada_header.nbeam) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting NBEAM, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "ZOOM_NCHAN", "%d", dada_header.zoom_nchan) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting ZOOM_NCHAN, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "PKT_HEADER", "%d", dada_header.pkt_header) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting PKT_HEADER, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

if (ascii_header_set(dada_header_buffer, "PKT_DATA", "%d", dada_header.pkt_data) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting PKT_DATA, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "PKT_NSAMP", "%d", dada_header.pkt_nsamp) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting PKT_NSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "PKT_NCHAN", "%d", dada_header.pkt_nchan) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting PKT_NCHAN, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "PKT_NPOL", "%d", dada_header.pkt_npol) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting PKT_NPOL, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "PKT_NBIT", "%d", dada_header.pkt_nbit) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting PKT_NBIT, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "PKT_TSAMP", "%f", dada_header.pkt_tsamp) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting PKT_TSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "NAVERAGE_BF", "%d", dada_header.naverage_bf) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting NAVERAGE_BF, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "NAVERAGE_ZOOM", "%d", dada_header.naverage_zoom) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting NAVERAGE_ZOOM, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  return EXIT_SUCCESS;
}
