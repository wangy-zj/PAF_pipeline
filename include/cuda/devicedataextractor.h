#ifndef _DEVICEDATAEXTRACTOR_H
#define _DEVICEDATAEXTRACTOR_H

#pragma once

#include "sharedutilities.h"

/*! \brief A class to copy device data to host
 *
 * \tparam T data type, which can be a complex data type
 * 
 */
template <typename T>
class DeviceDataExtractor {
public:
  T *data = NULL; ///< Host buffer to hold data
  
  /*!
   * \param[in] d_data Device data 
   * \param[in] ndata Number of data on host as type \p T
   * async memcpy will not work here as we always get new copy of memory
   */
  DeviceDataExtractor(T *d_data, int ndata)
    :d_data(d_data), ndata(ndata){
   
    size = ndata*sizeof(T);
    checkCudaErrors(cudaMallocHost(&data, size));
    checkCudaErrors(cudaMemcpy(data, d_data, size, cudaMemcpyDefault));
  }
  
  //! Deconstructor of DeviceDataExtractor class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~DeviceDataExtractor(){
    checkCudaErrors(cudaFreeHost(data));
  }
  
private:
  T *d_data = NULL;
  int ndata;
  int size;
};

#endif
