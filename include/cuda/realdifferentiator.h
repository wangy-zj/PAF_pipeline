#ifndef _REALDIFFERENTIATOR_H
#define _REALDIFFERENTIATOR_H

#pragma once

#include "sharedutilities.h"

/*! \brief Overloadded kernel to get d_difference between two real input vectors
 *
 * \tparam T1 Data type of the first input vector
 * \tparam T2 Data type of the second input vector
 * 
 * \param[in]  d_data1 The first input vector in \p T1
 * \param[in]  d_data2 The second input vector in \p T2
 * \param[in]  ndata   Number of data
 * \param[out] d_diff  The d_difference between these two vectors in float, it is always in float
 *
 * The kernel uses `scalar_subtract` to get difference (in float) between two numbers and currently it supports (we can add more support later).
 *
 * T1     | T2
 * -------|----
 * float  | float
 * float  | half
 * half   | float 
 * half   | half 
 * 
 * \see scalar_subtract
 * 
 */
template <typename T1, typename T2>
__global__ void real_subtract(const T1 *d_data1, const T2 *d_data2, float *d_diff, int ndata){
  int idx = blockDim.x*blockIdx.x + threadIdx.x;

  if(idx < ndata){
    //d_diff[idx] = d_data1[idx] - d_data2[idx];

    scalar_subtract(d_data1[idx], d_data2[idx], d_diff[idx]);
  }
}

/*! \brief A class to get the difference between two real vectors
 *
 * \tparam T1 Typename of the data in one vector
 * \tparam T2 Typename of the data in the other vector
 *
 * 
 * Suggested combinations of T1 and T2 are (other combinations may not work, we can add more support later)
 * T1     | T2
 * -------|----
 * float  | float
 * float  | half
 * half   | float 
 * half   | half  
 * 
 * The class to get difference between two real vectors, it is allowed to have different types for these inputs and
 * the result will be in float.
 * 
 */
template <typename T1, typename T2>
class RealDifferentiator {

public:
  float *data  = NULL;  ///< the difference between input \p data1 and \p data2
  
  //! Constructor of RealDifferentiator class.
  /*!
   * 
   * - initialise the class
   * - create device memory for the difference \p diff
   * - calculate the difference with a CUDA kernel `real_subtract`
   *
   * \see real_subtract, scalar_subtract
   * 
   * \param[in] raw1 The first input real vector
   * \param[in] raw2 The second input real vector
   * \param[in] ndata   Number of data to subtract
   * \param[in] nthread Number of threads per CUDA block to run `real_subtract`
   *
   */
  RealDifferentiator(T1 *raw1, T2 *raw2, int ndata, int nthread)
    :ndata(ndata), nthread(nthread){

    // sort out input buffers
    data1 = copy2device(raw1, ndata, type1);
    data2 = copy2device(raw2, ndata, type2);

    // Create output buffer as managed
    checkCudaErrors(cudaMallocManaged(&data, ndata*sizeof(float), cudaMemAttachGlobal));
    
    // setup kernel size and run it to get difference
    nblock = ceil(ndata/(float)nthread+0.5);
    real_subtract<<<nblock, nthread>>>(data1, data2, data, ndata);
    getLastCudaError("Kernel execution failed [ real_subtract ]");

    // Free intermediate memory
    remove_device_copy(type1, data1);
    remove_device_copy(type2, data2);
    
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  //! Deconstructor of RealDifferentiator class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~RealDifferentiator(){
    checkCudaErrors(cudaFree(data));
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
private:
  enum cudaMemoryType type1; ///< memory type
  enum cudaMemoryType type2; ///< memory type

  int ndata; ///< Number of input data
  int nthread; ///< Number of threads per CUDA block
  int nblock;  ///< Number of CUDA blocks

  T1 *data1 = NULL; ///< private variable to hold input vector pointer 1
  T2 *data2 = NULL; ///< private variable to hold input vector pointer 2
};

#endif
