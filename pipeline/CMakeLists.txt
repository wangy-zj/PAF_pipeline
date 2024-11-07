add_executable(pipeline_dada_beamform pipeline_dada_beamform.cu ${SOURCE_FILES})
target_include_directories(pipeline_dada_beamform PRIVATE $ENV{HOME}/.local/include)
target_link_libraries(pipeline_dada_beamform ${CUDA_LIBRARIES} ${PSRDADA_LIBRARY} OpenMP::OpenMP_CXX udppipe)
set_target_properties(pipeline_dada_beamform PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(dada_speed_test dada_speed_test.cu ${SOURCE_FILES})
target_include_directories(dada_speed_test PRIVATE $ENV{HOME}/.local/include)
target_link_libraries(dada_speed_test ${CUDA_LIBRARIES} ${PSRDADA_LIBRARY} OpenMP::OpenMP_CXX udppipe)
set_target_properties(dada_speed_test PROPERTIES LINKER_LANGUAGE CUDA)