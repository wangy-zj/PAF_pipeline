#ifndef _REALGENERATOR_H
#define _REALGENERATOR_H

#pragma once

#include "sharedutilities.h"


/*! A kernel to contraint random number from range (0.0 1.0] to range (exclude include] or [include exclude).
 *
 * \param[in, out] data    The input data in range (0.0 1.0] and new data in range (exclude include] or [include exclude) is also returned with it.
 * \param[in]      exclude The exclusive end of random numbers
 * \param[in]      range   The range of random numbers, it does not have to be positive, it is calculated with `include - exclude`
 * \param[in]      ndata   Number of data
 * \tparam         T       We do not really need it here, just to make the function templated so that I can put it into a header file without complain
 *
 */
template<typename T>
__global__ void real_contraintor(T *data, T exclude, T range, int ndata){
  // Maximum x-dimension of a grid of thread blocks is 2^31-1
  // Maximum x- or y-dimension of a block is 1024
  // So here we can cover (2^31-1)*1024 random numbers, which are 2^41-1024
  // should be big enough

  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(idx<ndata){
    // Just in case we have a very small ndata
    data[idx] = data[idx]*range+exclude;
  }
}
//__global__ void real_contraintor(float *data, float exclude, float range, int ndata);

/*! \brief A class to generate uniform distributed \p ndata random float data on device 
 *
 * The clase is created to generate uniform distributed \p ndata random float data on device in the range (exclude include] or [include exclude)
 * It uses `curandGenerateUniform` [curand](https://docs.nvidia.com/cuda/curand/index.html) API to generate random on device directly and then constraint it to a given range
 *
 */
class RealGeneratorUniform{
public:
  float *data = NULL; ///< Unified Memory to hold generated uniform distributed random numbers in float
  
  //! Constructor of RealGeneratorUniform class.
  /*!
   * 
   * - initialise the class
   * - create device memory for \p ndata float random numbers
   * - generate uniform distributed random numbers with `curandGenerateUniform` [curand](https://docs.nvidia.com/cuda/curand/index.html) API. 
   * 
   * `curandGenerateUniform` generates uniform distributed random numbers in the range of (0.0 1.0], the class converts the random numbers into a range defined by `exclude` and `include`, 
   * where `include` is the inclusive limit and `exclude` is the exclusive limit. If `exclude` is larger than `include`, the final data is in range [include exclude), otherwise the it is in range (exclude include]. If `include` is equal to `exclude`, we will get a constant number series as `exclude`. 
   * 
   * \param[in] gen     curand generator, should not create the generator inside the class, 
   *                    otherwise it is very likely that the same random numbers will be generated with different class instantiations
   * \param[in] exclude The exclusive limit of uniform random numbers
   * \param[in] include The inclusive limit if uniform random numbers
   * \param[in] ndata   Number of float random numbers to generate
   */
  RealGeneratorUniform(curandGenerator_t gen, int ndata, float exclude, float include, int nthread)
    :gen(gen), ndata(ndata), exclude(exclude), include(include), nthread(nthread){

    // Figure out range
    range = include-exclude;

    // Create output buffer as managed
    checkCudaErrors(cudaMallocManaged(&data, ndata*sizeof(float), cudaMemAttachGlobal));

    // Generate data
    checkCudaErrors(curandGenerateUniform(gen, data, ndata));

    // Setup kernel size and run it to convert to a given range
    nblock = ceil(ndata/(float)nthread+0.5);
    real_contraintor<float><<<nblock, nthread>>>(data, exclude, range, ndata);

    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  //! Deconstructor of RealGeneratorUniform class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~RealGeneratorUniform(){
    checkCudaErrors(cudaFree(data));
    checkCudaErrors(cudaDeviceSynchronize());
  }
    
private:
  int ndata;   ///< Number of generated data
  float include;  ///< inclusive limit of random numbers
  float exclude;  ///< inclusive limit of random numbers
  float range;    ///< Range
  int nthread;    ///< Number of threads
  int nblock;     ///< Number of cuda blocks
  
  curandGenerator_t gen; ///< Generator to generate uniform distributed random numbers
};

/*! \brief A class to generate normal distributed \p ndata random float data on device with given \p mean and \p stddev
 *
 * The clase is created to generate normal distributed \p ndata random float data on device with given \p mean and \p stddev.
 * It uses `curandGenerateNormal` [curand](https://docs.nvidia.com/cuda/curand/index.html) API to generate random on device directly, no further process happens here. 
 *
 */

class RealGeneratorNormal{
public:
  float *data = NULL; ///< Unified memory to hold normal distributed random numbers
  
  //! Constructor of RealGeneratorNormal class.
  /*!
   * 
   * - initialise the class
   * - create device memory for \p ndata float random numbers
   * - generate normal distributed random numbers with `curandGenerateNormal` [curand](https://docs.nvidia.com/cuda/curand/index.html) API. 
   * 
   * \param[in] gen    curand generator, should not create the generator inside the class, 
   *                   otherwise it is very likely that the same random numbers will be generated with different class instantiations
   * \param[in] mean   Required mean for normal distributed random numbers
   * \param[in] stddev Required standard deviation for normal distributed random numbers
   * \param[in] ndata  Number of float random numbers to generate
   */
  RealGeneratorNormal(curandGenerator_t gen, float mean, float stddev, int ndata)
    :gen(gen), mean(mean), stddev(stddev), ndata(ndata){

    // Create output buffer
    checkCudaErrors(cudaMallocManaged(&data, ndata*sizeof(float), cudaMemAttachGlobal));

    // Generate normal data
    checkCudaErrors(curandGenerateNormal(gen, data, ndata, mean, stddev));
  }
  
  //! Deconstructor of RealGeneratorNormal class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~RealGeneratorNormal(){
    checkCudaErrors(cudaFree(data));
  }
    
private:
  float mean;  ///< Mean of generated data
  float stddev;///< Standard deviation of generated data
  int ndata;   ///< Number of generated data
  
  curandGenerator_t gen; ///< Generator to generate normal distributed random numbers
};

#endif
