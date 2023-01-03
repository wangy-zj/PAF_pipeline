#ifndef _REALCONVERTOR_H
#define _REALCONVERTOR_H

#pragma once

#include "sharedutilities.h"

/*! \brief A function to convert real data from \p TIN to \p TOUT on GPU
 * 
 * It is a template function which converts data from \p TIN to \p TOUT, where
 * \tparam TIN  The input data type
 * \tparam TOUT The output data type
 *
 * The data type convertation is done with an overloadded function `scalar_typecast`
 * \see scalar_typecast
 *
 * The supported data convertation is shown in the following table (we can add more support here later)
 *
 * TIN    | TOUT
 * -------|----
 * float  | float
 * float  | double
 * float  | half 
 * float  | int
 * float  | int16_t
 * float  | int8_t
 * double | float
 * half   | float
 * int    | float
 * int16_t| float
 * int8_t | float
 *
 * \param[in]  input  Data to be converted
 * \param[in]  ndata  Number of data points to be converted
 * \param[out] output Converted data
 *
 */
template <typename TIN, typename TOUT>
__global__ void real_convertor(const TIN *input, TOUT *output, int ndata){
  // Maximum x-dimension of a grid of thread blocks is 2^31-1
  // Maximum x- or y-dimension of a block is 1024
  // So here we can cover (2^31-1)*1024 random numbers, which are 2^41-1024
  // should be big enough
  
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(idx<ndata){
    // Just in case we have a very small ndata
    scalar_typecast(input[idx], output[idx]);
  }
}

/*! \brief A class to convert real device data from one type \p TIN to another \p TOUT
 * 
 * It is a template class which converts data from \p TIN to \p TOUT, where
 * \tparam TIN  The input data type
 * \tparam TOUT The output data type
 *
 * The data type convertation is done with a template CUDA kernel function `real_convertor`
 * \see real_convertor
 *
 * The supported data convertation is shown in the following table (we can add more support here later)
 *
 * TIN    | TOUT
 * -------|----
 * float  | float
 * float  | double
 * float  | half 
 * float  | int
 * float  | int16_t
 * float  | int8_t
 * double | float
 * half   | float
 * int    | float
 * int16_t| float
 * int8_t | float
 *
 */
template <typename TIN, typename TOUT>
class RealConvertor{
public:
  TOUT *data = NULL; ///< Converted data on Unified memory
  
  //! Constructor of RealConvertor class.
  /*!
   * 
   * - initialise the class
   * - create device memory for \p ndata float random numbers
   * - convert input data from \p TIN to \p TOUT on GPU with CUDA Kernel `real_convertor`
   *
   * \see real_convertor
   *
   * \tparam TIN Input data type
   * 
   * \param[in] raw     Data to be converted with data type \p TIN on device or host
   * \param[in] ndata   Number of data points ton be converted
   * \param[in] nthread Number of threads per CUDA block to run `real_convertor` kernel
   *
   */
  RealConvertor(TIN *raw, int ndata, int nthread)
    :ndata(ndata), nthread(nthread){

    input = copy2device(raw, ndata, type);
    
    // Create output buffer as managed
    checkCudaErrors(cudaMallocManaged(&data, ndata*sizeof(TOUT), cudaMemAttachGlobal));

    // Setup kernel size and run it to convert data
    nblock = ceil(ndata/(float)nthread+0.5);
    real_convertor<<<nblock, nthread>>>(input, data, ndata);
    getLastCudaError("Kernel execution failed [ real_convertor ]");

    // Free intermediate memory
    remove_device_copy(type, input);
    
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  //! Deconstructor of RealConvertor class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~RealConvertor(){  
    checkCudaErrors(cudaFree(data));
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
private:
  enum cudaMemoryType type; ///< memory type
  TIN *input = NULL; ///< An internal pointer to input data
  int ndata;   ///< Number of generated data
  int nthread; ///< Number of threads per CUDA block
  int nblock;  ///< Number of blocks to process \p ndata
};

#endif
