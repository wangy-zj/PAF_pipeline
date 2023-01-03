#ifndef __DADA_HEADER_H
#define __DADA_HEADER_H

#define DADA_STRLEN 1024

#ifdef __cplusplus
extern "C" {
#endif

#include "inttypes.h"

  typedef struct dada_header_t{
    double mjd_start;
    int npkt;
    int nelement;
    int nbeam;
    int zoom_nchan;
    int pkt_header;
    int pkt_data;
    int pkt_nsamp;
    int pkt_nchan;
    int pkt_npol;
    int pkt_nbit;
    double pkt_tsamp;
    int naverage_bf;
    int naverage_zoom;
  }dada_header_t;

  int read_dada_header(const char *dada_header_buffer, dada_header_t *dada_header);

  int write_dada_header(const dada_header_t dada_header, char *dada_header_buffer);

  int read_dada_header_from_file(const char *dada_header_file_name, dada_header_t *dada_header);

  int write_dada_header_to_file(const dada_header_t dada_header,const char *dada_header_file_name);

#ifdef __cplusplus
}
#endif

#endif
