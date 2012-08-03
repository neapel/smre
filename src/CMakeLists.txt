
# Stick common SMRE stuff into a library, used by tests etc.
set(BASE_SRC
	io.cpp
		matlab_io.cpp
		plain_io.cpp
		gil_io.cpp
	opencl_helpers.cpp
	chambolle_pock.cpp
)

add_library(smre SHARED ${BASE_SRC})
require_libraries(smre)


# Executables using that library
add_executable(main  main.cpp)
target_link_libraries(main ${CMAKE_REQUIRED_LIBRARIES})
