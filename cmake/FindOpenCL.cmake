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
)

# Library
find_library(OpenCL_LIBRARY
	NAMES OpenCL
)

set(OpenCL_PROCESS_INCLUDES  OpenCL_INCLUDE_DIR OpenCL_INCLUDE_DIRS)
set(OpenCL_PROCESS_LIBS  OpenCL_LIBRARY OpenCL_LIBRARIES)
libfind_process(OpenCL)

