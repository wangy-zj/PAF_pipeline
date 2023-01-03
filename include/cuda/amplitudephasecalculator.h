#ifndef _AMPLITUDEPHASECALCULATOR_H
#define _AMPLITUDEPHASECALCULATOR_H

#pragma once

#include "sharedutilities.h"
//! A template kernel to calculate phase and amplitude of input array 
/*!
 * 
 * \see scalar_typecast
 *
 * \tparam T Complex number component data type
 * 
 * \param[in]  v         input Complex data
 * \param[in]  ndata     Number of data samples to be calculated
 * \param[out] amplitude Calculated amplitude
 * \param[out] phase     Calculated amplitude
 *
 */
template <typename T>
__global__ void amplitude_phase_calculator(const T *v, float *amplitude, float *phase, int ndata){
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  
  if(idx < ndata){
    // We always do calculation in float
    float v1;
    float v2;
    
    scalar_typecast(v[idx].x, v1);
    scalar_typecast(v[idx].y, v2);
    
    amplitude[idx] = sqrtf(v1*v1+v2*v2);
    phase[idx]     = atan2f(v2, v1); // in radians
  }
}

template <typename T>
class AmplitudePhaseCalculator{
public:
  float *amp = NULL;///< Calculated amplitude on device
  float *pha = NULL;///< Calculated phase on device
  
  //! Constructor of AmplitudePhaseCalculator class.
  /*!
   * 
   * - initialise the class
   * - create device memory for amplitude and phase
   * - calculate phase and amplitude with CUDA
   *
   * \see amplitude_phase_calculator
   *
   * \tparam TIN Input data type
   * 
   * \param[in] raw  input Complex data
   * \param[in] ndata   Number of samples to be converted, the size of data is 2*ndata
   * \param[in] nthread Number of threads per CUDA block to run `amplitude_phase_calculator` kernel
   *
   */
  AmplitudePhaseCalculator(T *raw,
			   int ndata,
			   int nthread
			   )
    :ndata(ndata), nthread(nthread){

    // sourt out input data
    data = copy2device(raw, ndata, type);
    
    // Get output buffer as managed
    checkCudaErrors(cudaMallocManaged(&amp, ndata * sizeof(float), cudaMemAttachGlobal));
    checkCudaErrors(cudaMallocManaged(&pha, ndata * sizeof(float), cudaMemAttachGlobal));
  
    // Get amplitude and phase
    nblock = ceil(ndata/(float)nthread+0.5);
    amplitude_phase_calculator<<<nblock, nthread>>>(data, amp, pha, ndata);

    remove_device_copy(type, data);
    
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  //! Deconstructor of RealGeneratorNormal class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~AmplitudePhaseCalculator(){
    
    checkCudaErrors(cudaFree(amp));
    checkCudaErrors(cudaFree(pha));

    checkCudaErrors(cudaDeviceSynchronize());
  }

private:
  int ndata; ///< number of values as a private parameter
  int nblock; ///< Number of CUDA blocks
  int nthread; ///< number of threas per block
  
  enum cudaMemoryType type; ///< memory type
  
  T *data; ///< To get hold on the input data
};


#endif
