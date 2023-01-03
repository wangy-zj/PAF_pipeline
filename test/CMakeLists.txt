# for GPU code
add_executable(test_unpack_1ant1pol test_unpack_1ant1pol.cu)
target_include_directories(test_unpack_1ant1pol PRIVATE $ENV{HOME}/.local/include)
target_link_libraries(test_unpack_1ant1pol ${CUDA_LIBRARIES} ${PSRDADA_LIBRARY} OpenMP::OpenMP_CXX udppipe)
set_target_properties(test_unpack_1ant1pol PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(test_power_taccumulate_1ant1pol test_power_taccumulate_1ant1pol.cu)
target_include_directories(test_power_taccumulate_1ant1pol PRIVATE $ENV{HOME}/.local/include)
target_link_libraries(test_power_taccumulate_1ant1pol ${CUDA_LIBRARIES} ${PSRDADA_LIBRARY} OpenMP::OpenMP_CXX udppipe)
set_target_properties(test_power_taccumulate_1ant1pol PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(test_tf2ft_1ant1pol test_tf2ft_1ant1pol.cu)
target_include_directories(test_tf2ft_1ant1pol PRIVATE $ENV{HOME}/.local/include)
target_link_libraries(test_tf2ft_1ant1pol ${CUDA_LIBRARIES} ${PSRDADA_LIBRARY} OpenMP::OpenMP_CXX udppipe)
set_target_properties(test_tf2ft_1ant1pol PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(test_rfft test_rfft.cu)
target_include_directories(test_rfft PRIVATE $ENV{HOME}/.local/include)
target_link_libraries(test_rfft ${CUDA_LIBRARIES} ${PSRDADA_LIBRARY} OpenMP::OpenMP_CXX udppipe)
set_target_properties(test_rfft PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(test_cufft test_cufft.cu)
target_include_directories(test_cufft PRIVATE $ENV{HOME}/.local/include)
target_link_libraries(test_cufft ${CUDA_LIBRARIES} ${PSRDADA_LIBRARY} OpenMP::OpenMP_CXX udppipe)
set_target_properties(test_cufft PROPERTIES LINKER_LANGUAGE CUDA)

