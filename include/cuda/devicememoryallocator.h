#ifndef _DEVICEMEMORYALLOCATOR_H
#define _DEVICEMEMORYALLOCATOR_H

#pragma once

#include "sharedutilities.h"

/*! \brief A class to allocate memory on device 
 *
 * \tparam T data type of the device memory, which can be a complex data type
 * 
 */
template <typename T>
class DeviceMemoryAllocator {
public:
  T *data = NULL; ///< Device memory or managed memory
  
  /*! Constructor of class DeviceMemoryAllocator
   *
   * \param[in] ndata Number of data on host as type \p T
   * \param[in] host  Marker to see if we also need a copy on host
   *
   */
  DeviceMemoryAllocator(int ndata, int host=0)
    :ndata(ndata), host(host){
    if(host){
      checkCudaErrors(cudaMallocManaged(&data, ndata*sizeof(T), cudaMemAttachGlobal));
    }
    else{
      checkCudaErrors(cudaMalloc(&data, ndata*sizeof(T)));
    }
  }
  
  //! Deconstructor of DeviceMemoryAllocator class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~DeviceMemoryAllocator(){
    checkCudaErrors(cudaFree(data));
  }

private:
  int ndata; ///< Number of data points
  int host;  ///< Do we also need to copy on host?
};


#endif
