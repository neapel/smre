
# Stick common SMRE stuff into a library, used by tests etc.
set(BASE_SRC
	io.cpp
		matlab_io.cpp
		plain_io.cpp
)
if(HAVE_OPENCL)
	set(BASE_SRC ${BASE_SRC}
		opencl_helpers.cpp
	)
endif()

add_library(smre SHARED ${BASE_SRC})
target_link_libraries(smre ${CMAKE_REQUIRED_LIBRARIES})

set(CMAKE_REQUIRED_LIBRARIES smre ${CMAKE_REQUIRED_LIBRARIES})


# Executables using that library
add_executable(main  main.cpp)
target_link_libraries(main ${CMAKE_REQUIRED_LIBRARIES})
