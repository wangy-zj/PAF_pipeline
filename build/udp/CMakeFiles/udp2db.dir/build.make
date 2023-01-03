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
include udp/CMakeFiles/udp2db.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include udp/CMakeFiles/udp2db.dir/compiler_depend.make

# Include the progress variables for this target.
include udp/CMakeFiles/udp2db.dir/progress.make

# Include the compile flags for this target's objects.
include udp/CMakeFiles/udp2db.dir/flags.make

udp/CMakeFiles/udp2db.dir/udp2db.cpp.o: udp/CMakeFiles/udp2db.dir/flags.make
udp/CMakeFiles/udp2db.dir/udp2db.cpp.o: /home/hero/code/udp-pipeline/udp/udp2db.cpp
udp/CMakeFiles/udp2db.dir/udp2db.cpp.o: udp/CMakeFiles/udp2db.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hero/code/udp-pipeline/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object udp/CMakeFiles/udp2db.dir/udp2db.cpp.o"
	cd /home/hero/code/udp-pipeline/build/udp && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT udp/CMakeFiles/udp2db.dir/udp2db.cpp.o -MF CMakeFiles/udp2db.dir/udp2db.cpp.o.d -o CMakeFiles/udp2db.dir/udp2db.cpp.o -c /home/hero/code/udp-pipeline/udp/udp2db.cpp

udp/CMakeFiles/udp2db.dir/udp2db.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/udp2db.dir/udp2db.cpp.i"
	cd /home/hero/code/udp-pipeline/build/udp && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hero/code/udp-pipeline/udp/udp2db.cpp > CMakeFiles/udp2db.dir/udp2db.cpp.i

udp/CMakeFiles/udp2db.dir/udp2db.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/udp2db.dir/udp2db.cpp.s"
	cd /home/hero/code/udp-pipeline/build/udp && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hero/code/udp-pipeline/udp/udp2db.cpp -o CMakeFiles/udp2db.dir/udp2db.cpp.s

# Object files for target udp2db
udp2db_OBJECTS = \
"CMakeFiles/udp2db.dir/udp2db.cpp.o"

# External object files for target udp2db
udp2db_EXTERNAL_OBJECTS =

udp/udp2db: udp/CMakeFiles/udp2db.dir/udp2db.cpp.o
udp/udp2db: udp/CMakeFiles/udp2db.dir/build.make
udp/udp2db: /usr/local/lib/libpsrdada.so
udp/udp2db: src/libudppipe.a
udp/udp2db: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
udp/udp2db: /usr/lib/x86_64-linux-gnu/libpthread.so
udp/udp2db: udp/CMakeFiles/udp2db.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hero/code/udp-pipeline/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable udp2db"
	cd /home/hero/code/udp-pipeline/build/udp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/udp2db.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
udp/CMakeFiles/udp2db.dir/build: udp/udp2db
.PHONY : udp/CMakeFiles/udp2db.dir/build

udp/CMakeFiles/udp2db.dir/clean:
	cd /home/hero/code/udp-pipeline/build/udp && $(CMAKE_COMMAND) -P CMakeFiles/udp2db.dir/cmake_clean.cmake
.PHONY : udp/CMakeFiles/udp2db.dir/clean

udp/CMakeFiles/udp2db.dir/depend:
	cd /home/hero/code/udp-pipeline/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hero/code/udp-pipeline /home/hero/code/udp-pipeline/udp /home/hero/code/udp-pipeline/build /home/hero/code/udp-pipeline/build/udp /home/hero/code/udp-pipeline/build/udp/CMakeFiles/udp2db.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : udp/CMakeFiles/udp2db.dir/depend

