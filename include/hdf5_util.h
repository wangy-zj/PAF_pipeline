#ifndef _HDF5_UTIL_H
#define _HDF5_UTIL_H

#include <stdlib.h>
#include <assert.h>
#include "hdf5.h"

#define H5FAIL -1

#ifdef __cplusplus
extern "C" {
#endif

  hid_t h5_create_file(char *h5fname);
  hid_t h5_create_dset(hid_t file_id, char *dset_name, hsize_t *dims, hsize_t *chunk_dims, hid_t dtype, int nrank);
  int h5_fill_dset(hid_t dset_id, hsize_t *offset, hsize_t *dimsext, hid_t dtype, int nrank, void *data);
  int h5_fill_attr(hid_t field_id, char *attr_name, hid_t dtype, void *attr_value);
#ifdef __cplusplus
}
#endif

#endif
