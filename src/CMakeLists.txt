# mangle CL code file into string, compile to object file.
set(CL_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cl_code.cpp)
set(CL_SRC chambolle_pock_kernels.cl)
add_custom_command(
	OUTPUT ${CL_OUTPUT}
	COMMAND ${CMAKE_SOURCE_DIR}/cmake/mangle_cl.sh ${CL_OUTPUT} ${CL_SRC}
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	DEPENDS ${CL_SRC}
	VERBATIM
)

# Stick common SMRE stuff into a library, used by tests etc.
set(BASE_SRC
	io.cpp
		matlab_io.cpp
		plain_io.cpp
		gil_io.cpp
	opencl_helpers.cpp
	chambolle_pock.cpp
		chambolle_pock_cl.cpp
		chambolle_pock_cpu.cpp
	${CL_OUTPUT}
)

if(BUILD_TESTING)
	add_library(smre SHARED ${BASE_SRC})
	#require_libraries(smre)
endif()


# Executables using that library
#add_executable(main main.cpp)
#target_link_libraries(main ${CMAKE_REQUIRED_LIBRARIES})

if(WITH_GUI)
	add_executable(guimain guimain.cpp ${BASE_SRC})
	target_link_libraries(guimain ${CMAKE_REQUIRED_LIBRARIES})
endif()
