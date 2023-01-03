#ifndef _HOSTMEMORYALLOCATOR_H
#define _HOSTMEMORYALLOCATOR_H

#pragma once

#include "sharedutilities.h"

/*! \brief A class to allocate memory on host 
 *
 * \tparam T data type of the host memory, which can be a complex data type
 * 
 */
template <typename T>
class HostMemoryAllocator {
public:
  T *data = NULL; ///< Host memory or managed memory
  
  /*! Constructor of class HostMemoryAllocator
   *
   * \param[in] ndata  Number of data on host as type \p T
   * \param[in] device Marker to tell if we also need a copy on device
   *
   */
  HostMemoryAllocator(int ndata, int device=0)
    :ndata(ndata), device(device){
    if(device){
      checkCudaErrors(cudaMallocManaged(&data, ndata*sizeof(T), cudaMemAttachGlobal));
    }
    else{
      checkCudaErrors(cudaMallocHost(&data, ndata*sizeof(T)));
    }
  }
  
  //! Deconstructor of HostMemoryAllocator class.
  /*!
   * 
   * - free host memory at the class life end
   */
  ~HostMemoryAllocator(){
    checkCudaErrors(cudaFreeHost(data));
    if(device){
      checkCudaErrors(cudaFree(data));
    }
  }

private:
  int ndata; ///< Number of data points
  int device; ///< Do we need a copy on device?
};


#endif
