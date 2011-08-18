cmake_minimum_required(VERSION 2.8)
project(SMRE)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

include_directories(
	"abstract"
	"include"
	"src"
)


# MatIO libraries
find_package(MatIO REQUIRED)
include_directories(${MatIO_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${MatIO_LIBRARIES})


# General compiler flags
add_definitions(-Wall -Wextra -g3)


# SVN
find_package(Subversion)
if(Subversion_FOUND)
	Subversion_WC_INFO(. ER)
	set(VERSION "SVN rev${ER_WC_REVISION}")
endif(Subversion_FOUND)



# config.h
configure_file(
	cmake/config.h.in
	src/config.h
)



# tests
enable_testing()



add_subdirectory(src)
add_subdirectory(test)
