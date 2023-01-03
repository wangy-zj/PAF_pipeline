#ifndef _DADA_UTIL_H
#define _DADA_UTIL_H

#include <stdlib.h>

#include "ipcio.h"
#include "futils.h"
#include "ipcbuf.h"
#include "dada_def.h"
#include "ascii_header.h"
#include "dada_hdu.h"
#include "multilog.h"

#include "dada_def.h"

#ifdef __cplusplus
extern "C" {
#endif

  int dada_verify_block_size(int nbytes_expected, ipcbuf_t *dblock);
  dada_hdu_t* dada_setup_hdu(key_t key, int read, multilog_t* log);
  
#ifdef __cplusplus
}
#endif

#endif
