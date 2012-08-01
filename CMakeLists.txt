cmake_minimum_required(VERSION 2.8)
project("SMRE" CXX C)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

include_directories(
	"abstract"
	"src"
)

macro(require_libraries LIBS)
	list(APPEND CMAKE_REQUIRED_LIBRARIES ${LIBS})
endmacro()



# Tests
include(CTest)
enable_testing()


# MatIO libraries
option(USE_MATIO "Build with Matlab datafile support" ON)
if(USE_MATIO)
	find_package(MatIO)
	if(MatIO_FOUND)
		include_directories(${MatIO_INCLUDE_DIRS})
		require_libraries(${MatIO_LIBRARIES})
		set(HAVE_MATIO ${MatIO_FOUND})
	endif()
endif()


# OpenCL libraries
option(USE_OPENCL "Build with OpenCL kernels" ON)
if(USE_OPENCL)
	find_package(OpenCL REQUIRED)
	if(OpenCL_FOUND)
		include_directories(${OpenCL_INCLUDE_DIRS})
		require_libraries(${OpenCL_LIBRARIES})
		set(HAVE_OPENCL ${OpenCL_FOUND})
	endif()

	# AMD FFT libraries
	find_package(AmdFFT REQUIRED)
	if(AmdFFT_FOUND)
		include_directories(${AmdFFT_INCLUDE_DIRS})
		require_libraries(${AmdFFT_LIBRARIES})
		set(HAVE_AMD_FFT ${AmdFFT_FOUND})
	endif()
endif()


# FFTw
find_package(FFTw REQUIRED)
if(FFTw_FOUND)
	include_directories(${FFTw_INCLUDE_DIRS})
	require_libraries(${FFTw_LIBRARIES})
endif()
find_package(FFTwf REQUIRED)
if(FFTwf_FOUND)
	include_directories(${FFTwf_INCLUDE_DIRS})
	require_libraries(${FFTwf_LIBRARIES})
endif()


# Boost libraries
find_package(Boost
	COMPONENTS program_options
	REQUIRED)
if(Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIRS})
	require_libraries(${Boost_LIBRARIES})
endif()

# Image libraries for use with Boost::GIL
option(USE_GIL "Build with Boost::GIL image io" ON)
if(USE_GIL)
	#find_package(JPEG) # unusable mess, uses exit() error handling.
	if(JPEG_FOUND)
		include_directories(${JPEG_INCLUDE_DIRS})
		require_libraries(${JPEG_LIBRARIES})
		set(HAVE_JPEG ${JPEG_FOUND})
	endif()

	find_package(PNG)
	if(PNG_FOUND)
		include_directories(${PNG_INCLUDE_DIRS})
		require_libraries(${PNG_LIBRARIES})
		set(HAVE_PNG ${PNG_FOUND})
		add_definitions(${PNG_DEFINITIONS})
	endif()

	find_package(TIFF)
	if(TIFF_FOUND)
		include_directories(${TIFF_INCLUDE_DIRS})
		require_libraries(${TIFF_LIBRARIES})
		set(HAVE_TIFF ${TIFF_FOUND})
	endif()
	set(HAVE_GIL  ${HAVE_JPEG} | ${HAVE_PNG} | ${HAVE_TIFF})
endif()


# Compiler specifics
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

macro(REQUIRE_SOURCE_COMPILES SOURCE VAR)
	check_cxx_source_compiles("${SOURCE}\n int main() {return 0;}" ${VAR})
	if(${VAR})
		message(STATUS "Compiler supports ${VAR}.")
	else()
		message(FATAL_ERROR "Compiler support test ${VAR} failed.")
	endif()
endmacro()

# General compiler flags
if(CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra --std=c++0x")
	add_definitions(-Wall -Wextra --std=c++0x)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g3")
	add_definitions(-O3 -g3)
endif()

require_source_compiles("
	void test() {
		auto k = 1 + 1;
		k = 0;
	}
" CXX_HAS_AUTO)
require_source_compiles("
	void test() {
		decltype(1 + 1) k = 1;
		k = 0;
	}
" CXX_HAS_DECLTYPE)
require_source_compiles("
	#include <vector>
	void test() {
		std::vector<int> i = {1, 2, 3};
		for(int &j : i) j = 0;
	}
" CXX_HAS_FOREACH)


# Profiling
option(USE_COVERAGE "Build with gcov and profiling support" OFF)
if(USE_COVERAGE)
	set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --coverage")
	set(CMAKE_LINKER_FLAGS  "${CMAKE_LINKER_FLAGS} --coverage")
	message(STATUS "Building with coverage and profiling support")
endif()


# SVN
find_package(Subversion)
if(Subversion_FOUND)
	Subversion_WC_INFO(${CMAKE_SOURCE_DIR} ER)
	set(VERSION  "SVN rev${ER_WC_REVISION}")
endif()


# Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
	set(DOXYGEN_CONF  ${CMAKE_CURRENT_BINARY_DIR}/doxygen.conf)
	set(DOXYGEN_SOURCE_DIR  "${CMAKE_SOURCE_DIR}/src")
	set(DOXYGEN_OUTPUT_DIR  "${CMAKE_CURRENT_BINARY_DIR}/doc")
	set(DOXYGEN_INCLUDE_PATH  "")
	string(REGEX REPLACE ";" " " DOXYGEN_INPUT_LIST "${DOXYGEN_SOURCE_DIR}")
	set(DOXYGEN_STRIP_FROM_PATH  "${DOXYGEN_INPUT_LIST}")
	configure_file(${CMAKE_SOURCE_DIR}/cmake/doxygen.conf.in ${DOXYGEN_CONF})

	add_custom_target(doc
		${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONF}
		DEPENDS ${DOXYGEN_CONF}
	)
endif()



# config.h
configure_file(${CMAKE_SOURCE_DIR}/cmake/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})



# Primary
add_subdirectory(src)


# Tests
if(BUILD_TESTING)
	add_subdirectory(test)
endif()
