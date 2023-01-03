#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../include/hdf5_util.h"

hid_t h5_create_file(char *h5fname){
  
  hid_t file_id = H5Fcreate (h5fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  H5Fclose(file_id);
  
  /* open it again with RDWR makes data writing faster*/
  return H5Fopen(h5fname, H5F_ACC_RDWR, H5P_DEFAULT);  
}

hid_t h5_create_dset(hid_t id, char *dset_name, hsize_t *dims, hsize_t *chunk_dims, hid_t dtype, int nrank){
  /* setup max dims, should be H5S_UNLIMITED*/
  hsize_t *maxdims = (hsize_t *)malloc(nrank*sizeof(hsize_t));
  for(int i = 0; i < nrank; i++){
    maxdims[i] = H5S_UNLIMITED;
    //maxdims[i] = dims[i]*2;
  }
  
  /* Create the data space for the dset. */
  hid_t dataspace_id = H5Screate_simple(nrank, dims, maxdims);
  free(maxdims);
  
  /* Modify dset creation properties, i.e. enable chunking  */
  hid_t prop    = H5Pcreate(H5P_DATASET_CREATE);
  herr_t status = H5Pset_chunk(prop, nrank, chunk_dims);
  assert(status!=H5FAIL);
  
  /* Create the dset. */
  hid_t dset_id =
    H5Dcreate2(id, dset_name, dtype, dataspace_id, H5P_DEFAULT, prop, H5P_DEFAULT);
  
  /* Close properties */
  status = H5Pclose(prop);
  assert(status != H5FAIL);
  
  /* Terminate access to the data space. */
  status = H5Sclose(dataspace_id);
  assert(status!=H5FAIL);

  /* return dset_id*/
  return dset_id;
}

int h5_fill_dset(hid_t dset_id, hsize_t *offset, hsize_t *dimsext, hid_t dtype, int nrank, void *data){

  hsize_t *size = (hsize_t *)malloc(nrank*sizeof(hsize_t));
  
  /* Get existing dset for offset */
  for(int i = 0; i < nrank; i++){
    size[i] = dimsext[i] + offset[i];
  }
  
  /* Extend its dims */
  herr_t status = H5Dset_extent(dset_id, size);
  assert(status!=H5FAIL);

  /* Select a hyperslab in extended portion of dset  */
  hid_t dataspace_id = H5Dget_space(dset_id); /* Need to get space again here */
  status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, dimsext, NULL);
  assert(status!=H5FAIL);
  
  /* Define memory space */
  hid_t memspace = H5Screate_simple(nrank, dimsext, NULL);
  
  /* Write the data to the extended portion of dset  */
  status = H5Dwrite(dset_id, dtype, memspace, dataspace_id, H5P_DEFAULT, data);
  assert(status!=H5FAIL);
  
  /* Terminate access to the data space. */
  status = H5Sclose(dataspace_id);
  assert(status!=H5FAIL);
  
  /* Close memory space */
  status = H5Sclose(memspace);
  assert(status!=H5FAIL);

  /* Free space */
  free(size);

  return EXIT_SUCCESS;
}

int h5_fill_attr(hid_t field_id, char *attr_name, hid_t dtype, void *attr_value){

  hid_t aid = H5Screate(H5S_SCALAR);
  hid_t attr = H5Acreate2(field_id, attr_name, dtype, aid, H5P_DEFAULT, H5P_DEFAULT);

  herr_t h5_status  = H5Awrite(attr, dtype, attr_value);
  assert(h5_status!=H5FAIL);

  h5_status = H5Aclose(attr);
  assert(h5_status!=H5FAIL);
  
  h5_status = H5Sclose(aid);
  assert(h5_status!=H5FAIL);

  return EXIT_SUCCESS;
}
