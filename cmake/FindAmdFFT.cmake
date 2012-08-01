# - Try to find the AmdFFT SDK
# Once done, this will define
#
#  AmdFFT_FOUND - system has it
#  AmdFFT_INCLUDE_DIRS - the include directories
#  AmdFFT_LIBRARIES - link these

include(FindPackageHandleStandardArgs)

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

# Version
set(AMDFFT_VERSION_FILE ${AMDFFT_INCLUDE_DIR}/clAmdFft.version.h)
if(EXISTS ${AMDFFT_VERSION_FILE})
	file(STRINGS ${AMDFFT_VERSION_FILE} AMDFFT_MAJOR REGEX "#define[ ]+clAmdFftVersionMajor[ ]+[0-9]+")
	file(STRINGS ${AMDFFT_VERSION_FILE} AMDFFT_MINOR REGEX "#define[ ]+clAmdFftVersionMinor[ ]+[0-9]+")
	file(STRINGS ${AMDFFT_VERSION_FILE} AMDFFT_PATCH REGEX "#define[ ]+clAmdFftVersionPatch[ ]+[0-9]+")
	string(REGEX MATCH "[0-9]+" AMDFFT_MAJOR ${AMDFFT_MAJOR})
	string(REGEX MATCH "[0-9]+" AMDFFT_MINOR ${AMDFFT_MINOR})
	string(REGEX MATCH "[0-9]+" AMDFFT_PATCH ${AMDFFT_PATCH})
	set(AMDFFT_VERSION "${AMDFFT_MAJOR}.${AMDFFT_MINOR}.${AMDFFT_PATCH}")
endif()

find_package_handle_standard_args(AMDFFT
	REQUIRED_VARS AMDFFT_INCLUDE_DIR AMDFFT_LIBRARY
	VERSION_VAR AMDFFT_VERSION)
