cmake_minimum_required(VERSION 2.8)
project("SMRE" CXX)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

include_directories(
	"abstract"
	"src"
)

macro(require_libraries LIBS)
	set(CMAKE_REQUIRED_LIBRARIES  ${CMAKE_REQUIRED_LIBRARIES} ${LIBS})
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
		set(HAVE_MATIO  ${MatIO_FOUND})
	endif()
endif()


# FFTw
find_package(FFTw REQUIRED)
if(FFTw_FOUND)
	include_directories(${FFTw_INCLUDE_DIRS})
	require_libraries(${FFTw_LIBRARIES})
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


# General compiler flags
add_definitions(-Wall -Wextra -g3)

# C++11
add_definitions(--std=c++0x)


# Profiling
option(USE_COVERAGE "Build with gcov and profiling support" ON)
if(USE_COVERAGE)
	set(CMAKE_CXX_FLAGS  ${CMAKE_CXX_FLAGS}  "--coverage")
	set(CMAKE_LINKER_FLAGS  ${CMAKE_LINKER_FLAGS}  "--coverage")
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
	STRING(REGEX REPLACE ";" " " DOXYGEN_INPUT_LIST "${DOXYGEN_SOURCE_DIR}")
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
