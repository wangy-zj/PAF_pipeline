#ifndef _HOSTDATAEXTRACTOR_H
#define _HOSTDATAEXTRACTOR_H

#pragma once

#include "sharedutilities.h"

/*! \brief A class to copy data from host to device
 *
 * \tparam T data type, which can be a complex data type
 * 
 */
template <typename T>
class HostDataExtractor {
public:
  T *data = NULL; ///< Device buffer to hold data

  /*! Constructor of class HostDataExtractor
   *
   * \param[in] h_data Host data 
   * \param[in] ndata Number of data on host as type \p T
   * async memcpy will not work here as we always get new copy of memory
   */
  HostDataExtractor(T *h_data, int ndata)
    :h_data(h_data), ndata(ndata){
    
    size = ndata*sizeof(T);
    checkCudaErrors(cudaMalloc(&data, size));
    checkCudaErrors(cudaMemcpy(data, h_data, size, cudaMemcpyDefault));
  }
  
  //! Deconstructor of HostDataExtractor class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~HostDataExtractor(){
    checkCudaErrors(cudaFree(data));
  }
  
private:
  T *h_data = NULL;
  int ndata;
  int size;
};

#endif
