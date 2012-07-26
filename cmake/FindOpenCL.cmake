# - Try to find an OpenCL SDK
# Once done, this will define
#
#  OpenCL_FOUND - system has it
#  OpenCL_INCLUDE_DIRS - the include directories
#  OpenCL_LIBRARIES - link these

include(LibFindMacros)

# Include dir
find_path(OpenCL_INCLUDE_DIR
	NAMES CL/cl.h
	HINTS ${AMD_APP_SDK_ROOT}/include
	ENV AMD_APP_SDK_ROOT
)

# Library
find_library(OpenCL_LIBRARY
	NAMES OpenCL
	HINTS ${AMD_APP_SDK_ROOT}/lib
	ENV AMD_APP_SDK_ROOT
	PATH_SUFFIXES x86_64 x86
)

set(OpenCL_PROCESS_INCLUDES  OpenCL_INCLUDE_DIR OpenCL_INCLUDE_DIRS)
set(OpenCL_PROCESS_LIBS  OpenCL_LIBRARY OpenCL_LIBRARIES)
libfind_process(OpenCL)

