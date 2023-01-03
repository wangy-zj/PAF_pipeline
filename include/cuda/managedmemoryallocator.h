#ifndef _MANAGEDMEMORYALLOCATOR_H
#define _MANAGEDMEMORYALLOCATOR_H

#pragma once

#include "sharedutilities.h"

/*! \brief A class to allocate memory as managed
 *
 * \tparam T data type of the host memory, which can be a complex data type
 * 
 */
template <typename T>
class ManagedMemoryAllocator {
public:
  T *data = NULL; ///< Managed memory 
  
  /*! Constructor of class ManagedMemoryAllocator
   *
   * \param[in] ndata  Number of data on host as type \p T
   *
   */
  ManagedMemoryAllocator(int ndata)
    :ndata(ndata){
    checkCudaErrors(cudaMallocManaged(&data, ndata*sizeof(T), cudaMemAttachGlobal));
  }
  
  //! Deconstructor of ManagedMemoryAllocator class.
  /*!
   * 
   * - free host memory at the class life end
   */
  ~ManagedMemoryAllocator(){
    checkCudaErrors(cudaFree(data));
  }
  
private:
  int ndata; ///< Number of data points
};


#endif
