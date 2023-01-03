#ifndef _COMPLEXSPLITTER_H
#define _COMPLEXSPLITTER_H

#pragma once

#include "sharedutilities.h"

//! A template kernel to split complex numbers into its real and imag part
/*!
 * 
 * \see scalar_typecast
 *
 * \tparam TREAL Real part data type
 * \tparam TIMAG Imag part data type
 * \tparam TCMPX Complex data type
 * 
 * \param[in]  d_cmpx  Complex numbers 
 * \param[out] d_real  Real part of complex numbers
 * \param[out] d_imag  Imag part of complex numbers
 * \param[in]  ndata   Number of data points to be splitted
 *
 */
template <typename TCMPX, typename TREAL, typename TIMAG>
__global__ void complex_splitter(const TCMPX *d_cmpx, const TREAL *d_real, TIMAG *d_imag, int ndata){
  // Maximum x-dimension of a grid of thread blocks is 2^31-1
  // Maximum x- or y-dimension of a block is 1024
  // So here we can cover (2^31-1)*1024 random numbers, which are 2^41-1024
  // should be big enough
  
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(idx<ndata){

    scalar_typecast(d_cmpx[idx].x,   d_real[idx]);
    scalar_typecast(d_cmpx[idx].y, d_imag[idx]);
  }
}

/*! \brief A class to build a complex vector with two real vectors
 *
 * \tparam TREAL Typename of the real part data
 * \tparam TIMAG Typename of the imag part data
 * \tparam TCMPX Typename of the complex data
 *
 * The class use kernel `cudautil_complexbuilder` to convert data type and build complex numbers. `cudautil_complexbuilder` uses `scalar_typecast` to convert data type. 
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
template <typename TCMPX, typename TREAL, typename TIMAG>
class ComplexSplitter {
public:
  TREAL *real = NULL; ///< Real part on device
  TIMAG *imag = NULL; ///< Imag part on device
  
  //! Constructor of ComplexSplitter class.
  /*!
   * 
   * - initialise the class
   * - create device memory for \p ndata real and imag numbers
   * - split complex numbers to \p real and \p imag
   *
   * \see cudautil_complexbuilder, scalar_typecast
   *
   * \tparam TREAL Real part data type
   * \tparam TIMAG Imag part data type
   * \tparam TCMPX Complex data type
   * 
   * \param[in] cmpx  Complex numbers
   * \param[in] ndata   Number of data points ton be converted
   * \param[in] nthread Number of threads per CUDA block to run `cudautil_complexbuilder` kernel
   *
   */
  
  ComplexSplitter(TCMPX *cmpx, int ndata, int nthread)
    :ndata(ndata), nthread(nthread){

    // Sort out input buffer
    data = copy2device(cmpx, ndata, type);
    
    // Create managed memory for output
    checkCudaErrors(cudaMallocManaged(&real, ndata*sizeof(TREAL), cudaMemAttachGlobal));
    checkCudaErrors(cudaMallocManaged(&imag, ndata*sizeof(TIMAG), cudaMemAttachGlobal));

    // Setup kernel and run it 
    nblock = ceil(ndata/(float)nthread+0.5);
    
    complex_splitter<<<nblock, nthread>>>(data, real, imag, ndata);
    getLastCudaError("Kernel execution failed [ complex_splitter ]");

    // Free intermediate memory
    remove_device_copy(type, data);    
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  //! Deconstructor of ComplexSplitter class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~ComplexSplitter(){
    checkCudaErrors(cudaFree(real));
    checkCudaErrors(cudaFree(imag));

    checkCudaErrors(cudaDeviceSynchronize());
  }
  
private:
  TCMPX *data = NULL;
  
  enum cudaMemoryType type; ///< memory type
    
  int ndata;
  int nthread;
  int nblock;
};

#endif
