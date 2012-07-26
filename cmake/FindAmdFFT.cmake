# - Try to find the AmdFFT SDK
# Once done, this will define
#
#  AmdFFT_FOUND - system has it
#  AmdFFT_INCLUDE_DIRS - the include directories
#  AmdFFT_LIBRARIES - link these

include(LibFindMacros)

# Include dir
find_path(AmdFFT_INCLUDE_DIR
	NAMES clAmdFft.h
	HINTS ${AMD_FFT_ROOT}/include
	ENV AMD_FFT_ROOT
)

# Library
find_library(AmdFFT_LIBRARY
	NAMES clAmdFft.Runtime
	HINTS ${AMD_FFT_ROOT}/lib64 ${AMD_FFT_ROOT}/lib32 
	ENV AMD_FFT_ROOT
	PATH_SUFFIXES x86_64 x86
)

set(AmdFFT_PROCESS_INCLUDES  AmdFFT_INCLUDE_DIR AmdFFT_INCLUDE_DIRS)
set(AmdFFT_PROCESS_LIBS  AmdFFT_LIBRARY AmdFFT_LIBRARIES)
libfind_process(AmdFFT)


