# - Try to find the AmdFFT SDK
# Once done, this will define
#
#  AMDFFT_FOUND - system has it
#  AMDFFT_INCLUDE_DIRS - the include directories
#  AMDFFT_LIBRARIES - link these

# Need OpenCL 1.0
find_package(OpenCL REQUIRED)


include(FindPackageHandleStandardArgs)

# Find AMD FFT SDK root
find_path(AMDFFT_ROOT_DIR
	NAMES include/clAmdFft.h
	HINTS ${AMDFFT_ROOT_DIR}
	ENV AMDFFTROOT
)

# Include dir
find_path(AMDFFT_INCLUDE_DIR
	NAMES clAmdFft.h
	HINTS ${AMDFFT_ROOT_DIR}/include
)

# Library
find_library(AMDFFT_LIBRARY
	NAMES clAmdFft.Runtime
	HINTS ${AMDFFT_ROOT_DIR}/lib64 ${AMDFFT_ROOT_DIR}/lib32 
)

# Compile version test
if(AMDFFT_INCLUDE_DIR AND AMDFFT_LIBRARY)
	set(_AMDFFT_TEST_SOURCE "
		#include <clAmdFft.h>
		#include <stdio.h>
		#include <stdlib.h>

		int main() {
			cl_uint major, minor, patch;
			if(clAmdFftGetVersion(&major, &minor, &patch) == CL_SUCCESS) {
				printf(\"%d.%d.%d\", major, minor, patch);
				fflush(stdout);
				return EXIT_SUCCESS;
			}
			return EXIT_FAILURE;
		}
	")
	set(_AMDFFT_TEST_FILE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/amdfftversion.c")
	file(WRITE ${_AMDFFT_TEST_FILE} "${_AMDFFT_TEST_SOURCE}\n")

	try_run(_AMDFFT_RUN_RESULT _AMDFFT_COMPILE_RESULT
		${CMAKE_BINARY_DIR} ${_AMDFFT_TEST_FILE}
		COMPILE_OUTPUT_VARIABLE _AMDFFT_COMPILE_OUTPUT
		RUN_OUTPUT_VARIABLE AMDFFT_VERSION
		CMAKE_FLAGS
			"-DINCLUDE_DIRECTORIES:STRING=${AMDFFT_INCLUDE_DIR};${OPENCL_INCLUDE_DIRS}"
			"-DLINK_LIBRARIES:STRING=${AMDFFT_LIBRARY};${OPENCL_LIBRARIES}")
		if("${AMDFFT_VERSION}" STREQUAL "")
			message(FATAL_ERROR "Could not determine AmdFFT version: ${_AMDFFT_COMPILE_OUTPUT}")
		endif()
endif()

find_package_handle_standard_args(AMDFFT
	REQUIRED_VARS AMDFFT_INCLUDE_DIR AMDFFT_LIBRARY
	VERSION_VAR AMDFFT_VERSION)
