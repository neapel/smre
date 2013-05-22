cmake_minimum_required(VERSION 2.8)
project("SMRE" CXX C)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

include_directories("src" "src/vexcl/")

macro(require_libraries)
	list(APPEND CMAKE_REQUIRED_LIBRARIES ${ARGN})
endmacro()

# OpenCL libraries
find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIR})
require_libraries(${OPENCL_LIBRARY})

# FFTw
find_package(FFTwf REQUIRED)
include_directories(${FFTwf_INCLUDE_DIRS})
require_libraries(${FFTwf_LIBRARIES})

# Multiprecision
find_package(MPFR REQUIRED)
include_directories(${MPFR_INCLUDES})
require_libraries(${MPFR_LIBRARIES})

# Boost libraries
find_package(Boost
	COMPONENTS regex filesystem system chrono program_options
	REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
require_libraries(${Boost_LIBRARIES})

# GTK+3 GUI
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTKMM REQUIRED gtkmm-3.0)
include_directories(${GTKMM_INCLUDE_DIRS})
link_directories(${GTKMM_LIBRARY_DIRS})
require_libraries(${GTKMM_LIBRARIES})
add_definitions(${GTKMM_CFLAGS_OTHER})


# General compiler flags
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -Wall -Werror -Wuninitialized -Wmaybe-uninitialized -Winit-self")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3")
	set(HAVE_OPENMP 1)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
	set(HAVE_OPENMP 0)
endif()


# Profiling
option(USE_COVERAGE "Build with gcov and profiling support" OFF)
if(USE_COVERAGE)
	set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --coverage")
	set(CMAKE_LINKER_FLAGS  "${CMAKE_LINKER_FLAGS} --coverage")
	message(STATUS "Building with coverage and profiling support")
endif()


# config.h
configure_file(${CMAKE_SOURCE_DIR}/cmake/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Primary
add_subdirectory(src)

# Tests
add_subdirectory(test)
