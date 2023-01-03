file(GLOB SRCS CONFIGURE_DEPENDS "*.cu" "*.cpp" "*.c")
add_library(udppipe STATIC ${SRCS})
