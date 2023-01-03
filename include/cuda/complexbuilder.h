#ifndef _COMPLEXGENERATOR_H
#define _COMPLEXGENERATOR_H

#pragma once

#include "sharedutilities.h"

//! A template kernel to build complex numbers with its real and imag part
/*!
 * 
 * \see scalar_typecast
 *
 * \tparam TREAL Real part data type
 * \tparam TIMAG Imag part data type
 * \tparam TCMPX Complex data type
 * 
 * \param[in]  d_real  Real part to build complex numbers
 * \param[in]  d_imag  Imag part to build complex numbers
 * \param[in]  ndata   Number of data points ton be built
 * \param[out] d_cmpx  Complex numbers
 *
 */
template <typename TREAL, typename TIMAG, typename TCMPX>
__global__ void complex_builder(const TREAL *d_real, const TIMAG *d_imag, TCMPX *d_cmpx, int ndata){
  // Maximum x-dimension of a grid of thread blocks is 2^31-1
  // Maximum x- or y-dimension of a block is 1024
  // So here we can cover (2^31-1)*1024 random numbers, which are 2^41-1024
  // should be big enough
  
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(idx<ndata){

    scalar_typecast(d_real[idx], d_cmpx[idx].x);
    scalar_typecast(d_imag[idx], d_cmpx[idx].y);
  }
}

/*! \brief A class to build a complex vector with two real vectors
 *
 * \tparam TREAL Typename of the real part data
 * \tparam TIMAG Typename of the imag part data
 * \tparam TCMPX Typename of the complex data
 *
 * The class use kernel `complex_builder` to convert data type and build complex numbers. `complex_builder` uses `scalar_typecast` to convert data type. 
 * As of that, the allowed data type here is limited to following table (more types can be added later) 
 * 
 * TREAL/TIMAG    | TCMPX
 * -------|----
 * float  | cuComplex
 * float  | cuDoubleComplex
 * float  | half2
 * float  | int2
 * float  | short2
 * float  | int8_t ???
 * double | cuComplex
 * half   | cuComplex
 * int    | cuComplex
 * int16_t| cuComplex
 * int8_t | cuComplex
 *
 */
template <typename TREAL, typename TIMAG, typename TCMPX>
class ComplexBuilder {
public:
  TCMPX *data = NULL; ///< Complex data on device
  
  //! Constructor of ComplexBuilder class.
  /*!
   * 
   * - initialise the class
   * - create device memory for \p ndata complex numbers
   * - build complex numbers with \p real and \p imag
   *
   * \see complex_builder, scalar_typecast
   *
   * \tparam TREAL Real part data type
   * \tparam TIMAG Imag part data type
   * \tparam TCMPX Complex data type
   * 
   * \param[in] real  Real part to build complex numbers
   * \param[in] imag  Imag part to build complex numbers
   * \param[in] ndata   Number of data points ton be converted
   * \param[in] nthread Number of threads per CUDA block to run `complex_builder` kernel
   *
   */
  ComplexBuilder(TREAL *real, TIMAG *imag, int ndata, int nthread )
    :ndata(ndata), nthread(nthread){

    // Sort out input data
    data_real = copy2device(real, ndata, type_real);
    data_imag = copy2device(imag, ndata, type_imag);

    // Create output buffer
    checkCudaErrors(cudaMallocManaged(&data, ndata*sizeof(TCMPX), cudaMemAttachGlobal));

    // Setup kernel size and run it
    nblock = ceil(ndata/(float)nthread+0.5);
    complex_builder<<<nblock, nthread>>>(data_real, data_imag, data, ndata);
    getLastCudaError("Kernel execution failed [ complex_builder ]");

    // Free intermediate memory
    remove_device_copy(type_real, data_real);
    remove_device_copy(type_imag, data_imag);

    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  //! Deconstructor of ComplexBuilder class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~ComplexBuilder(){
    checkCudaErrors(cudaFree(data));
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
private:
  TREAL *data_real = NULL;
  TIMAG *data_imag = NULL;

  enum cudaMemoryType type_real; ///< memory type
  enum cudaMemoryType type_imag; ///< memory type
  
  int ndata;
  int nthread;
  int nblock;
};
#endif
