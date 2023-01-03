cmake_minimum_required (VERSION 3.13)

project (udp-pipeline LANGUAGES CUDA CXX C)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "61;75;86")
endif()

find_package(OpenMP)

set(CUDA_LIBRARIES cufft curand cudart cublas)
find_library(PSRDADA_LIBRARY psrdada $ENV{PSRDADA_ROOT}/lib)

add_subdirectory(src)
add_subdirectory(udp)
add_subdirectory(test)
add_subdirectory(pipeline)
#add_subdirectory(dada)

