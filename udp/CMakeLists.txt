add_executable(udpgen udpgen.cpp udp.cpp)
target_include_directories(udpgen PRIVATE $ENV{DIFXROOT}/include $ENV{HOME}/.local/include)
target_link_libraries(udpgen ${PSRDADA_LIBRARY} OpenMP::OpenMP_CXX udppipe)

add_executable(udp2db udp2db.cpp udp.cpp)
target_include_directories(udp2db PRIVATE $ENV{DIFXROOT}/include $ENV{HOME}/.local/include)
target_link_libraries(udp2db ${PSRDADA_LIBRARY} OpenMP::OpenMP_CXX udppipe)
