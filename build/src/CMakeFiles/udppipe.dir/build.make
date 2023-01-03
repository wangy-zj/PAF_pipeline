# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hero/code/udp-pipeline

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hero/code/udp-pipeline/build

# Include any dependencies generated for this target.
include src/CMakeFiles/udppipe.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/udppipe.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/udppipe.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/udppipe.dir/flags.make

src/CMakeFiles/udppipe.dir/dada_header.c.o: src/CMakeFiles/udppipe.dir/flags.make
src/CMakeFiles/udppipe.dir/dada_header.c.o: /home/hero/code/udp-pipeline/src/dada_header.c
src/CMakeFiles/udppipe.dir/dada_header.c.o: src/CMakeFiles/udppipe.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hero/code/udp-pipeline/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/CMakeFiles/udppipe.dir/dada_header.c.o"
	cd /home/hero/code/udp-pipeline/build/src && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/udppipe.dir/dada_header.c.o -MF CMakeFiles/udppipe.dir/dada_header.c.o.d -o CMakeFiles/udppipe.dir/dada_header.c.o -c /home/hero/code/udp-pipeline/src/dada_header.c

src/CMakeFiles/udppipe.dir/dada_header.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/udppipe.dir/dada_header.c.i"
	cd /home/hero/code/udp-pipeline/build/src && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hero/code/udp-pipeline/src/dada_header.c > CMakeFiles/udppipe.dir/dada_header.c.i

src/CMakeFiles/udppipe.dir/dada_header.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/udppipe.dir/dada_header.c.s"
	cd /home/hero/code/udp-pipeline/build/src && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hero/code/udp-pipeline/src/dada_header.c -o CMakeFiles/udppipe.dir/dada_header.c.s

src/CMakeFiles/udppipe.dir/dada_util.c.o: src/CMakeFiles/udppipe.dir/flags.make
src/CMakeFiles/udppipe.dir/dada_util.c.o: /home/hero/code/udp-pipeline/src/dada_util.c
src/CMakeFiles/udppipe.dir/dada_util.c.o: src/CMakeFiles/udppipe.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hero/code/udp-pipeline/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object src/CMakeFiles/udppipe.dir/dada_util.c.o"
	cd /home/hero/code/udp-pipeline/build/src && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/udppipe.dir/dada_util.c.o -MF CMakeFiles/udppipe.dir/dada_util.c.o.d -o CMakeFiles/udppipe.dir/dada_util.c.o -c /home/hero/code/udp-pipeline/src/dada_util.c

src/CMakeFiles/udppipe.dir/dada_util.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/udppipe.dir/dada_util.c.i"
	cd /home/hero/code/udp-pipeline/build/src && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hero/code/udp-pipeline/src/dada_util.c > CMakeFiles/udppipe.dir/dada_util.c.i

src/CMakeFiles/udppipe.dir/dada_util.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/udppipe.dir/dada_util.c.s"
	cd /home/hero/code/udp-pipeline/build/src && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hero/code/udp-pipeline/src/dada_util.c -o CMakeFiles/udppipe.dir/dada_util.c.s

src/CMakeFiles/udppipe.dir/hdf5_util.c.o: src/CMakeFiles/udppipe.dir/flags.make
src/CMakeFiles/udppipe.dir/hdf5_util.c.o: /home/hero/code/udp-pipeline/src/hdf5_util.c
src/CMakeFiles/udppipe.dir/hdf5_util.c.o: src/CMakeFiles/udppipe.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hero/code/udp-pipeline/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object src/CMakeFiles/udppipe.dir/hdf5_util.c.o"
	cd /home/hero/code/udp-pipeline/build/src && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/udppipe.dir/hdf5_util.c.o -MF CMakeFiles/udppipe.dir/hdf5_util.c.o.d -o CMakeFiles/udppipe.dir/hdf5_util.c.o -c /home/hero/code/udp-pipeline/src/hdf5_util.c

src/CMakeFiles/udppipe.dir/hdf5_util.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/udppipe.dir/hdf5_util.c.i"
	cd /home/hero/code/udp-pipeline/build/src && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hero/code/udp-pipeline/src/hdf5_util.c > CMakeFiles/udppipe.dir/hdf5_util.c.i

src/CMakeFiles/udppipe.dir/hdf5_util.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/udppipe.dir/hdf5_util.c.s"
	cd /home/hero/code/udp-pipeline/build/src && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hero/code/udp-pipeline/src/hdf5_util.c -o CMakeFiles/udppipe.dir/hdf5_util.c.s

src/CMakeFiles/udppipe.dir/krnl.cu.o: src/CMakeFiles/udppipe.dir/flags.make
src/CMakeFiles/udppipe.dir/krnl.cu.o: /home/hero/code/udp-pipeline/src/krnl.cu
src/CMakeFiles/udppipe.dir/krnl.cu.o: src/CMakeFiles/udppipe.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hero/code/udp-pipeline/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object src/CMakeFiles/udppipe.dir/krnl.cu.o"
	cd /home/hero/code/udp-pipeline/build/src && /usr/local/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT src/CMakeFiles/udppipe.dir/krnl.cu.o -MF CMakeFiles/udppipe.dir/krnl.cu.o.d -x cu -c /home/hero/code/udp-pipeline/src/krnl.cu -o CMakeFiles/udppipe.dir/krnl.cu.o

src/CMakeFiles/udppipe.dir/krnl.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/udppipe.dir/krnl.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

src/CMakeFiles/udppipe.dir/krnl.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/udppipe.dir/krnl.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

src/CMakeFiles/udppipe.dir/test.cpp.o: src/CMakeFiles/udppipe.dir/flags.make
src/CMakeFiles/udppipe.dir/test.cpp.o: /home/hero/code/udp-pipeline/src/test.cpp
src/CMakeFiles/udppipe.dir/test.cpp.o: src/CMakeFiles/udppipe.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hero/code/udp-pipeline/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/udppipe.dir/test.cpp.o"
	cd /home/hero/code/udp-pipeline/build/src && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/udppipe.dir/test.cpp.o -MF CMakeFiles/udppipe.dir/test.cpp.o.d -o CMakeFiles/udppipe.dir/test.cpp.o -c /home/hero/code/udp-pipeline/src/test.cpp

src/CMakeFiles/udppipe.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/udppipe.dir/test.cpp.i"
	cd /home/hero/code/udp-pipeline/build/src && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hero/code/udp-pipeline/src/test.cpp > CMakeFiles/udppipe.dir/test.cpp.i

src/CMakeFiles/udppipe.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/udppipe.dir/test.cpp.s"
	cd /home/hero/code/udp-pipeline/build/src && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hero/code/udp-pipeline/src/test.cpp -o CMakeFiles/udppipe.dir/test.cpp.s

src/CMakeFiles/udppipe.dir/udp.cpp.o: src/CMakeFiles/udppipe.dir/flags.make
src/CMakeFiles/udppipe.dir/udp.cpp.o: /home/hero/code/udp-pipeline/src/udp.cpp
src/CMakeFiles/udppipe.dir/udp.cpp.o: src/CMakeFiles/udppipe.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hero/code/udp-pipeline/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/CMakeFiles/udppipe.dir/udp.cpp.o"
	cd /home/hero/code/udp-pipeline/build/src && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/udppipe.dir/udp.cpp.o -MF CMakeFiles/udppipe.dir/udp.cpp.o.d -o CMakeFiles/udppipe.dir/udp.cpp.o -c /home/hero/code/udp-pipeline/src/udp.cpp

src/CMakeFiles/udppipe.dir/udp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/udppipe.dir/udp.cpp.i"
	cd /home/hero/code/udp-pipeline/build/src && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hero/code/udp-pipeline/src/udp.cpp > CMakeFiles/udppipe.dir/udp.cpp.i

src/CMakeFiles/udppipe.dir/udp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/udppipe.dir/udp.cpp.s"
	cd /home/hero/code/udp-pipeline/build/src && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hero/code/udp-pipeline/src/udp.cpp -o CMakeFiles/udppipe.dir/udp.cpp.s

# Object files for target udppipe
udppipe_OBJECTS = \
"CMakeFiles/udppipe.dir/dada_header.c.o" \
"CMakeFiles/udppipe.dir/dada_util.c.o" \
"CMakeFiles/udppipe.dir/hdf5_util.c.o" \
"CMakeFiles/udppipe.dir/krnl.cu.o" \
"CMakeFiles/udppipe.dir/test.cpp.o" \
"CMakeFiles/udppipe.dir/udp.cpp.o"

# External object files for target udppipe
udppipe_EXTERNAL_OBJECTS =

src/libudppipe.a: src/CMakeFiles/udppipe.dir/dada_header.c.o
src/libudppipe.a: src/CMakeFiles/udppipe.dir/dada_util.c.o
src/libudppipe.a: src/CMakeFiles/udppipe.dir/hdf5_util.c.o
src/libudppipe.a: src/CMakeFiles/udppipe.dir/krnl.cu.o
src/libudppipe.a: src/CMakeFiles/udppipe.dir/test.cpp.o
src/libudppipe.a: src/CMakeFiles/udppipe.dir/udp.cpp.o
src/libudppipe.a: src/CMakeFiles/udppipe.dir/build.make
src/libudppipe.a: src/CMakeFiles/udppipe.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hero/code/udp-pipeline/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX static library libudppipe.a"
	cd /home/hero/code/udp-pipeline/build/src && $(CMAKE_COMMAND) -P CMakeFiles/udppipe.dir/cmake_clean_target.cmake
	cd /home/hero/code/udp-pipeline/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/udppipe.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/udppipe.dir/build: src/libudppipe.a
.PHONY : src/CMakeFiles/udppipe.dir/build

src/CMakeFiles/udppipe.dir/clean:
	cd /home/hero/code/udp-pipeline/build/src && $(CMAKE_COMMAND) -P CMakeFiles/udppipe.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/udppipe.dir/clean

src/CMakeFiles/udppipe.dir/depend:
	cd /home/hero/code/udp-pipeline/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hero/code/udp-pipeline /home/hero/code/udp-pipeline/src /home/hero/code/udp-pipeline/build /home/hero/code/udp-pipeline/build/src /home/hero/code/udp-pipeline/build/src/CMakeFiles/udppipe.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/udppipe.dir/depend

