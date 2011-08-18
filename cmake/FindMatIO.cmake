# - Try to find MatIO
# Once done, this will define
#
#  MatIO_FOUND - system has it
#  MatIO_INCLUDE_DIRS - the include directories
#  MatIO_LIBRARIES - link these

include(LibFindMacros)
libfind_pkg_check_modules(MatIO_PKGCONF matio)

# Include dir
find_path(MatIO_INCLUDE_DIR
	NAMES matio.h
	PATHS ${MatIO_PKGCONF_INCLUDE_DIRS}
)

# Library
find_library(MatIO_LIBRARY
	NAMES matio
	PATHS ${MatIO_PKGCONF_LIBRARY_DIRS}
)

set(MatIO_PROCESS_INCLUDES  MatIO_INCLUDE_DIR MatIO_INCLUDE_DIRS)
set(MatIO_PROCESS_LIBS  MatIO_LIBRARY MatIO_LIBRARIES)
libfind_process(MatIO)
