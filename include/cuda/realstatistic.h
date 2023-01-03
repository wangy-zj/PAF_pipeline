#ifndef _REALSTATISTIC_H
#define _REALSTATISTIC_H

#pragma once

#include "sharedutilities.h"

/*! \brief A function to convert input data from \p T to float and calculate its power in parallel on GPU
 * 
 * It is a template function to convert input data from \p T to float and calculate its power in parallel on GPU
 * \tparam T  The input data type
 *
 * The data type convertation is done with an overloadded function `scalar_typecast`
 * \see scalar_typecast
 *
 * The supported data convertation is shown in the following table (we can add more support here later)
 *
 * |T      |
 * |-------|
 * |double |
 * |half   |
 * |int    |
 * |int16_t|
 * |int8_t |
 *
 * \param[in]  d_data   Input data
 * \param[in]  ndata    Number of data
 * \param[out] d_float  Converted data in float
 * \param[out] d_float2 Power of converted data 
 *
 */
template <typename T>
__global__ void real_pow2(const T *d_data, float *d_float, float *d_float2, int ndata){
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  
  if(idx < ndata){
    float f_data;

    scalar_typecast(d_data[idx], f_data);
    d_float[idx]  = f_data;
    d_float2[idx] = f_data*f_data;
  }
}

template <typename T>
class RealMeanStddevCalculator {
  
public:
  float mean;   ///< Mean of the difference between input two vectors, always in float
  float stddev; ///< Standard deviation of the difference between input two vectors, always in float

  
  //! Constructor of class RealMeanStddevCalculator
  /*!
   * - initialise the class
   * - create required device memory
   * - convert input data from \p T to float and calculate its power in a single CUDA kernel `real_pow2`
   * - reduce the float data and its power to get mean
   * - calculate standard deviation with the mean of float and power data
   *
   * \param[in] raw     The input vector on device/host with data type \p T
   * \param[in] ndata   Number of data
   * \param[in] nthread Number of threads per CUDA block to run kernel `real_pow2`
   * \param[in] method  Data reduction method, which can be from 0 to 7 inclusive
   *
   * As kernel `real_pow2` uses `scalar_typecast` to convert \p T to float, the support \p T can be
   *
   * |T |
   * |--|
   * |double | 
   * |half   |
   * |int    | 
   * |int16_t|
   * |int8_t |
   * 
   * \see real_pow2, reduce, scalar_typecast
   *
   */
  RealMeanStddevCalculator(T *raw, int ndata, int nthread, int method)
    :ndata(ndata), nthread(nthread), method(method){

    /* Sort out input buffers */
    data = copy2device(raw, ndata, type);
    
    // Now do calculation
    nblock = ceil(ndata/(float)nthread+0.5);
    
    checkCudaErrors(cudaMallocManaged(&d_float,  ndata*sizeof(float), cudaMemAttachGlobal));
    checkCudaErrors(cudaMallocManaged(&d_float2, ndata*sizeof(float), cudaMemAttachGlobal));
    
    checkCudaErrors(cudaMallocManaged(&d_reduction, nblock*sizeof(float), cudaMemAttachGlobal));
    
    real_pow2<<<nblock, nthread>>>(data, d_float, d_float2, ndata);
    getLastCudaError("Kernel execution failed [ real_pow2 ]");
    
    // First reduce mean data
    reduce(ndata,  nthread, nblock, method, d_float, d_reduction);
    checkCudaErrors(cudaDeviceSynchronize());
    if(nblock > 1){
      reduce(nblock, nthread, 1, method, d_reduction, d_float);
      checkCudaErrors(cudaDeviceSynchronize());
      mean = d_float[0]/(float)ndata;
    }else{
      mean = d_reduction[0]/(float)ndata;
    }
    
    // Second reduce mean power 2 data
    reduce(ndata,  nthread, nblock, method, d_float2, d_reduction);
    checkCudaErrors(cudaDeviceSynchronize());
    if(nblock > 1){
      reduce(nblock, nthread, 1, method, d_reduction, d_float2);
      checkCudaErrors(cudaDeviceSynchronize());
      mean2 = d_float2[0]/(float)ndata;
    }else{
      mean2 = d_reduction[0]/(float)ndata;
    }

    // Got final numbers
    stddev = sqrtf(mean2 - mean*mean);

    // As we only need stddev and mean
    // Probably better to free all memory here
    checkCudaErrors(cudaFree(d_float));
    checkCudaErrors(cudaFree(d_float2));
    checkCudaErrors(cudaFree(d_reduction));
    
    remove_device_copy(type, data);
    
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  //! Deconstructor of RealMeanStddevCalculator class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~RealMeanStddevCalculator(){
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
private:
  enum cudaMemoryType type; ///< memory type

  int ndata; ///< Number of input data
  int nthread; ///< Number of threads per CUDA block
  int nblock;  ///< Number of CUDA blocks
  int method; ///< data d_reduction method
  
  T *data = NULL;
  float *d_float = NULL;
  float *d_float2 = NULL;

  float *d_reduction; ///< it holds intermediate float data duration data d_reduction on device
  
  float mean2; ///< mean of difference power of 2
};


#endif
