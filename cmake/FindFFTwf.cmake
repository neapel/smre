# - Try to find FFTw3f
# Once done, this will define
#
#  FFTw_FOUND - system has it
#  FFTw_INCLUDE_DIRS - the include directories
#  FFTw_LIBRARIES - link these

include(LibFindMacros)
libfind_pkg_check_modules(FFTwf_PKGCONF fftw3f)

# Include dir
find_path(FFTwf_INCLUDE_DIR
	NAMES fftw3.h
	PATHS ${FFTwf_PKGCONF_INCLUDE_DIRS}
)

# Library
find_library(FFTwf_LIBRARY
	NAMES fftw3f
	PATHS ${FFTwf_PKGCONF_LIBRARY_DIRS}
)

set(FFTwf_PROCESS_INCLUDES  FFTwf_INCLUDE_DIR)
set(FFTwf_PROCESS_LIBS  FFTwf_LIBRARY)
libfind_process(FFTwf)
