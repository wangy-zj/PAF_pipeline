// cuda_utilities.h
//
// last-edit-by: <> 
//
// Description:
//
//////////////////////////////////////////////////////////////////////

#ifndef _SHAREDUTILITIES_H
#define _SHAREDUTILITIES_H

#pragma once

#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <cufft.h>
#include <curand.h>
#include <curand_kernel.h>

#include "helper_cuda.h"

#define NUM_BINS 256
#define CUDA_STARTTIME(x)  cudaEventRecord(x ## _start, 0);
#define CUDA_STOPTIME(x) {					\
    float dtime;						\
    cudaEventRecord(x ## _stop, 0);				\
    cudaEventSynchronize(x ## _stop);				\
    cudaEventElapsedTime(&dtime, x ## _start, x ## _stop);	\
    x ## time += dtime; }

inline int print_cuda_memory_info() {
  //cudaError_t status;
  size_t free, total;
  
  checkCudaErrors(cudaMemGetInfo(&free, &total));
  
  fprintf(stdout, "GPU free memory is %.1f, total is %.1f MBbytes\n",
	  free/1024.0/1024, total/1024.0/1024);
  
  if(free<=0){
    fprintf(stderr, "Use too much GPU memory.\n");
    exit(EXIT_FAILURE);
  }
  
  return EXIT_SUCCESS;
}

/*! Overload cout with cuda complex data type
 */
#include <iostream>
static inline std::ostream& operator<<(std::ostream& os, const cuComplex& data){
  os << data.x << ' ' << data.y << ' ';
  return os;
}

/*! Overload * operator to multiple a cuComplex with a float for device and host code
 *
 * \param[in] a Input cuComplex number
 * \param[in] b Input float number
 * \returns   \a a * \a b
 *
 */
__device__ __host__ static inline cuComplex operator*(cuComplex a, float b) { return make_cuComplex(a.x*b, a.y*b);}

/*! Overload * operator to multiple a float with a cuComplex for device and host code
 *
 * \param[in] a Input float number
 * \param[in] b Input cuComplex number
 * \returns   \a a * \a b
 *
 */
__device__ __host__ static inline cuComplex operator*(float a, cuComplex b) { return make_cuComplex(b.x*a, b.y*a);}

/*! Overload / operator to divide a cuComplex with a float for device and host code
 *
 * \param[in, out] a A cuComplex number which will be divided by float \a b 
 * \param[in]      b Input float number
 * \returns        \a a / \a b
 *
 */
__device__ __host__ static inline cuComplex operator/(cuComplex a, float b) { return make_cuComplex(a.x/b, a.y/b);}

/*! Overload /= operator to divide a cuComplex with a float before it is accumulated to itself for device and host code
 *
 * \param[in, out] a A cuComplex number which will be divided by float \a b and accumulated to
 * \param[in]      b Input float number
 *
 */
__device__ __host__ static inline void operator/=(cuComplex &a, float b)     { a.x/=b;   a.y/=b;}

/*! Overload /= operator to plus a cuComplex with a cuComplex before it is accumulated to itself for device and host code
 *
 * \param[in, out] a A cuComplex number which will be added by cuComplex \a b and accumulated to
 * \param[in]      b Input cuComplex number
 *
 */
__device__ __host__ static inline void operator+=(cuComplex &a, cuComplex b) { a.x+=b.x; a.y+=b.y;}

/*! Overload /= operator to minus a cuComplex with a cuComplex before it is accumulated to itself for device and host code
 *
 * \param[in, out] a A cuComplex number which will be minused by cuComplex \a b and accumulated to
 * \param[in]      b Input cuComplex number
 *
 */
__device__ __host__ static inline void operator-=(cuComplex &a, cuComplex b) { a.x-=b.x; a.y-=b.y;}


#define CUDAUTIL_FLOAT2HALF __float2half
#define CUDAUTIL_FLOAT2INT  __float2int_rz
#define CUDAUTIL_FLOAT2UINT __float2uint_rz
#define CUDAUTIL_HALF2FLOAT __half2float
#define CUDAUTIL_DOUBLE2INT __double2int_rz

// We need more type case overload functions here
// The following convert float to other types
__device__ static inline void scalar_typecast(const float a, double   &b) { b = a;}
__device__ static inline void scalar_typecast(const float a, float    &b) { b = a;}
__device__ static inline void scalar_typecast(const float a, half     &b) { b = CUDAUTIL_FLOAT2HALF(a);}
__device__ static inline void scalar_typecast(const float a, int      &b) { b = CUDAUTIL_FLOAT2INT(a);}
__device__ static inline void scalar_typecast(const float a, int16_t  &b) { b = CUDAUTIL_FLOAT2INT(a);}
__device__ static inline void scalar_typecast(const float a, int8_t   &b) { b = CUDAUTIL_FLOAT2INT(a);}
__device__ static inline void scalar_typecast(const float a, unsigned &b) { b = CUDAUTIL_FLOAT2UINT(a);}

// The following convert other types to float
__device__ static inline void scalar_typecast(const double a,   float &b) { b = a;}
__device__ static inline void scalar_typecast(const half a,     float &b) { b = CUDAUTIL_HALF2FLOAT(a);}
__device__ static inline void scalar_typecast(const int a,      float &b) { b = a;}
__device__ static inline void scalar_typecast(const int16_t a,  float &b) { b = a;}
__device__ static inline void scalar_typecast(const int8_t a,   float &b) { b = a;}
__device__ static inline void scalar_typecast(const unsigned a, float &b) { b = a;}

template <typename TMIN, typename TSUB, typename TRES>
__device__ static inline void scalar_subtract(const TMIN minuend, const TSUB subtrahend, TRES &result) {
  TRES casted_minuend;
  TRES casted_subtrahend;
  
  scalar_typecast(minuend,    casted_minuend);
  scalar_typecast(subtrahend, casted_subtrahend);
  
  result = casted_minuend - casted_subtrahend;
}

template <typename TREAL, typename TIMAG, typename TCMPX>
__device__ static inline void make_cuComplex(const TREAL x, const TIMAG y, TCMPX &z){
  scalar_typecast(x, z.x);
  scalar_typecast(y, z.y);
}

/*! A template function to get a buffer into device
  It check where is the input buffer, if the buffer is on device, it just pass the pointer
  otherwise it alloc new buffer on device and copy the data to it
*/
template <typename T>
T* copy2device(T *raw, int ndata, enum cudaMemoryType &type){
  T *data = NULL;
  
  cudaPointerAttributes attributes; ///< to hold memory attributes
  
  // cudaMemoryTypeUnregistered for unregistered host memory,
  // cudaMemoryTypeHost for registered host memory,
  // cudaMemoryTypeDevice for device memory or
  // cudaMemoryTypeManaged for managed memory.
  checkCudaErrors(cudaPointerGetAttributes(&attributes, raw));
  type = attributes.type;
  
  if(type == cudaMemoryTypeUnregistered || type == cudaMemoryTypeHost){
    int nbytes = ndata*sizeof(T);
    checkCudaErrors(cudaMallocManaged(&data, nbytes, cudaMemAttachGlobal));
    checkCudaErrors(cudaMemcpy(data, raw, nbytes, cudaMemcpyDefault));
  }
  else{
    data = raw;
  }
  
  return data;
}

/*! A function to free memory if it is a copy of a host memory
 */
template<typename T>
int remove_device_copy(enum cudaMemoryType type, T *data){
  
  if(type == cudaMemoryTypeUnregistered || type == cudaMemoryTypeHost){
    checkCudaErrors(cudaFree(data));
  }
  
  return EXIT_SUCCESS;
}

#endif // CUDA_UTILITIES_H
//////////////////////////////////////////////////////////////////////
// $Log:$
//
