# - Try to find FFTw3
# Once done, this will define
#
#  FFTw_FOUND - system has it
#  FFTw_INCLUDE_DIRS - the include directories
#  FFTw_LIBRARIES - link these

include(LibFindMacros)
libfind_pkg_check_modules(FFTw_PKGCONF fftw3)

# Include dir
find_path(FFTw_INCLUDE_DIR
	NAMES fftw3.h
	PATHS ${FFTw_PKGCONF_INCLUDE_DIRS}
)

# Library
find_library(FFTw_LIBRARY
	NAMES fftw3
	PATHS ${FFTw_PKGCONF_LIBRARY_DIRS}
)

set(FFTw_PROCESS_INCLUDES  FFTw_INCLUDE_DIR FFTw_INCLUDE_DIRS)
set(FFTw_PROCESS_LIBS  FFTw_LIBRARY FFTw_LIBRARIES)
libfind_process(FFTw)

