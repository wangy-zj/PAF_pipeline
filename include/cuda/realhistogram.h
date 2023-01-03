#ifndef _REALHISTOGRAM_H
#define _REALHISTOGRAM_H

#pragma once

#include "sharedutilities.h"

#define NUM_BINS 256

/*! This kernel is not used for real input data, 
 *
 * \param[in] in  Data to be binned
 * \param[in] ndata Number of data points
 * \param[in] max   Maximum to check, not Maximum of given data
 * \param[in] max   Maximum to check, not Maximum of given data
 * \param[in] out   Binned data
 *
 * \tparam TIN  The input data type
 */
template <typename T>
__global__ void histogram(const T *in, int ndata, float min, float max, unsigned int *out)
{
  // pixel coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  // grid dimensions
  int nx = blockDim.x * gridDim.x; 

  // initialize temporary accumulation array in shared memory
  // has one extra element?
  // if blockDim.x is smaller than NUM_BINS, we have to process multiple bins per thread
  __shared__ unsigned int smem[NUM_BINS + 1];
  for (int i = threadIdx.x; i < NUM_BINS + 1; i += blockDim.x) smem[i] = 0;
  __syncthreads();

  // process pixels
  // updates our block's partial histogram in shared memory
  // if kernel size is smaller than data size, each thread may have to process multiple inputs
  for (int col = x; col < ndata; col += nx) {
    int r = ((in[col] - min)/(max-min))*NUM_BINS;
    if(r >= 0 && r < NUM_BINS){
      // ignore samples outside range
      atomicAdd(&smem[r], 1);
    }
  }
  __syncthreads();

  // write partial histogram into the global memory
  out += blockIdx.x * NUM_BINS;
  for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    out[i] = smem[i];
  }
}

/*! A kernel to finish binning process started by histogram_smem_atomics
  
  the kernel size here should be the same as previous gridDim.x size or larger
  
  \tparam T It is not necessary here but I need it to make the function templated.
  \see histogram
*/
template <typename T>
__global__ void histogram_final(const T *in, int n, unsigned int *out)
{
  // gridDim.x and blockDim.x should be enough to cover al bin index
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NUM_BINS) {
    T total = 0;
    for (int j = 0; j < n; j++) 
      total += in[i + NUM_BINS * j];
    out[i] = total;
  }
}

/*! \brief A class to get histogram of \p ndata input data with a data type \tparam T
 *
 */
template <typename T>
class RealHistogram{
public:
  unsigned *data = NULL; ///< Unified Memory to hold generated uniform distributed random numbers in float
  
  //! Constructor of RealHistogram class.
  /*!
   */
  RealHistogram(T *raw, int ndata, float min, float max, int nblock, int nthread)
    :ndata(ndata), min(min), max(max), nblock(nblock), nthread(nthread){

    // Sort out input data
    input = copy2device(raw, ndata, type);
    
    // Create output buffer as managed
    checkCudaErrors(cudaMallocManaged(&data, NUM_BINS*sizeof(unsigned), cudaMemAttachGlobal));

    // Setup kernel size and run it to get bin
    dim3 grid_smem(nblock);  // Do not want have too many blocks
    dim3 grid_final(nblock/nthread);
    
    checkCudaErrors(cudaMallocManaged(&result, nblock*NUM_BINS*sizeof(unsigned), cudaMemAttachGlobal));
    
    histogram<<<grid_smem, nthread>>>(input, ndata, min, max, result);
    getLastCudaError("Kernel execution failed [ histogram ]");
    histogram_final<unsigned int ><<<grid_final, nthread>>>(result, nblock, data);
    getLastCudaError("Kernel execution failed [ histogram_final ]");

    // free intermediate data
    remove_device_copy(type, input);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  //! Deconstructor of RealHistogram class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~RealHistogram(){
    checkCudaErrors(cudaFree(data));
    checkCudaErrors(cudaDeviceSynchronize());
  }
    
private:
  enum cudaMemoryType type; ///< memory type

  T *input; ///< input buffer
  int ndata;   ///< Number of generated data
  float min;  ///< min to check
  float max;  ///< max to check
  int nthread;    ///< Number of threads
  int nblock;     ///< Number of cuda blocks

  unsigned *result; ///< it is not the final result
};

#endif
