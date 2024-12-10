#add_executable(pipeline_dada_beamform pipeline_dada_beamform.cu ${SOURCE_FILES})
#target_include_directories(pipeline_dada_beamform PRIVATE $ENV{HOME}/.local/include)
#target_link_libraries(pipeline_dada_beamform ${CUDA_LIBRARIES} ${PSRDADA_LIBRARY} OpenMP::OpenMP_CXX udppipe)
#set_target_properties(pipeline_dada_beamform PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(dada_cputogpu_test dada_cputogpu_test.cu ${SOURCE_FILES})
target_include_directories(dada_cputogpu_test PRIVATE $ENV{HOME}/.local/include)
target_link_libraries(dada_cputogpu_test ${CUDA_LIBRARIES} ${PSRDADA_LIBRARY} OpenMP::OpenMP_CXX udppipe)
set_target_properties(dada_cputogpu_test PROPERTIES LINKER_LANGUAGE CUDA)


add_executable(dada_gputocpu_test dada_gputocpu_test.cu ${SOURCE_FILES})
target_include_directories(dada_gputocpu_test PRIVATE $ENV{HOME}/.local/include)
target_link_libraries(dada_gputocpu_test ${CUDA_LIBRARIES} ${PSRDADA_LIBRARY} OpenMP::OpenMP_CXX udppipe)
set_target_properties(dada_gputocpu_test PROPERTIES LINKER_LANGUAGE CUDA)


add_executable(dada_clear dada_clear.cu ${SOURCE_FILES})
target_include_directories(dada_clear PRIVATE $ENV{HOME}/.local/include)
target_link_libraries(dada_clear ${CUDA_LIBRARIES} ${PSRDADA_LIBRARY} OpenMP::OpenMP_CXX udppipe)
set_target_properties(dada_clear PROPERTIES LINKER_LANGUAGE CUDA)