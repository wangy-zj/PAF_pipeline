add_executable(test_dada_header test_dada_header.c dada_header.c)
target_include_directories(test_dada_header PRIVATE ${INCLUDE_ROOT})
target_link_libraries(test_dada_header m ${PSRDADA_LIBRARY})
