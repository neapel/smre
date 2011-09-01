cmake_minimum_required(VERSION 2.8)
project("SMRE" CXX)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

include_directories(
	"abstract"
	"include"
	"src"
)


# Tests
include(CTest)
enable_testing()


# MatIO libraries
option(USE_MATIO "Build with Matlab datafile support" ON)
if(USE_MATIO)
	find_package(MatIO)
	if(MatIO_FOUND)
		include_directories(${MatIO_INCLUDE_DIRS})
		set(CMAKE_REQUIRED_LIBRARIES  ${CMAKE_REQUIRED_LIBRARIES} ${MatIO_LIBRARIES})
		set(HAVE_MATIO  ${MatIO_FOUND})
	endif()
endif()

# General compiler flags
add_definitions(-Wall -Wextra -g3)


# Profiling
option(USE_COVERAGE "Build with gcov and profiling support" ON)
if(USE_COVERAGE)
	set(CMAKE_CXX_FLAGS  ${CMAKE_CXX_FLAGS}  "--coverage")
	set(CMAKE_LINKER_FLAGS  ${CMAKE_LINKER_FLAGS}  "--coverage")
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
	set(DOXYGEN_SOURCE_DIR  src abstract)
	set(DOXYGEN_OUTPUT_DIR  "doc")
	set(DOXYGEN_INCLUDE_PATH  "include")
	STRING(REGEX REPLACE ";" " " DOXYGEN_INPUT_LIST "${DOXYGEN_SOURCE_DIR}")
	configure_file(cmake/doxygen.conf.in ${DOXYGEN_CONF})

	add_custom_target(doc
		${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONF}
		DEPENDS ${DOXYGEN_CONF}
	)
endif()



# config.h
configure_file(cmake/config.h.in src/config.h)


add_subdirectory(src)
if(BUILD_TESTING)
	add_subdirectory(test)
endif()
