# - Try to find the AmdFFT SDK
# Once done, this will define
#
#  AmdFFT_FOUND - system has it
#  AmdFFT_INCLUDE_DIRS - the include directories
#  AmdFFT_LIBRARIES - link these

include(FindPackageHandleStandardArgs)

find_path(AmdFFT_ROOT_DIR
	NAMES include/clAmdFft.h
	HINTS ${AmdFFT_ROOT_DIR}
	ENV AMDFFTROOT
)

# Include dir
find_path(AmdFFT_INCLUDE_DIR
	NAMES clAmdFft.h
	HINTS ${AmdFFT_ROOT_DIR}/include
)

# Library
find_library(AmdFFT_LIBRARIES
	NAMES clAmdFft.Runtime
	HINTS ${AmdFFT_ROOT_DIR}/lib64 ${AmdFFT_ROOT_DIR}/lib32 
)

# Version
set(AmdFFT_VERSION_FILE ${AmdFFT_INCLUDE_DIR}/clAmdFft.version.h)
if(EXISTS ${AmdFFT_VERSION_FILE})
	file(STRINGS ${AmdFFT_VERSION_FILE} AmdFFT_MAJOR REGEX "#define[ ]+clAmdFftVersionMajor[ ]+[0-9]+")
	file(STRINGS ${AmdFFT_VERSION_FILE} AmdFFT_MINOR REGEX "#define[ ]+clAmdFftVersionMinor[ ]+[0-9]+")
	file(STRINGS ${AmdFFT_VERSION_FILE} AmdFFT_PATCH REGEX "#define[ ]+clAmdFftVersionPatch[ ]+[0-9]+")
	string(REGEX MATCH "[0-9]+" AmdFFT_MAJOR ${AmdFFT_MAJOR})
	string(REGEX MATCH "[0-9]+" AmdFFT_MINOR ${AmdFFT_MINOR})
	string(REGEX MATCH "[0-9]+" AmdFFT_PATCH ${AmdFFT_PATCH})
	set(AmdFFT_VERSION "${AmdFFT_MAJOR}.${AmdFFT_MINOR}.${AmdFFT_PATCH}")
endif()

find_package_handle_standard_args(AmdFFT
	REQUIRED_VARS AmdFFT_INCLUDE_DIR AmdFFT_LIBRARIES
	VERSION_VAR AmdFFT_VERSION)
