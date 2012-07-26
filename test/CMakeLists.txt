
# Test harness
set(CMAKE_REQUIRED_LIBRARIES smre ${CMAKE_REQUIRED_LIBRARIES})

macro(tiny_test NAME)
	add_executable(${NAME} "${NAME}.cpp")
	target_link_libraries(${NAME} ${CMAKE_REQUIRED_LIBRARIES})
	add_test(
		NAME ${NAME}
		COMMAND ${NAME} ${ARGN}
	)
	add_dependencies(${NAME} smre)
endmacro()

macro(extra_file NAME)
	configure_file(${NAME} ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
endmacro()

set(MAT_DIR "${CMAKE_SOURCE_DIR}/matfiles")

# Tests: return 0 for success.



# multi_array <-> uBLAS adaptors
tiny_test(test_multi_array_matrix)
tiny_test(test_multi_array_vector)

# multi_array tools
tiny_test(test_multi_array_operators)
tiny_test(test_multi_array_print)

# multi_array <-> FFTw adaptors
tiny_test(test_multi_array_fft)

# operator split
tiny_test(test_tuple_cut)
tiny_test(test_operator_split)

# io
if(HAVE_MATIO)
	tiny_test(test_matlab_io "${MAT_DIR}/test.mat")
endif()
tiny_test(test_plain_io)

if(HAVE_GIL)
	tiny_test(test_gil_io "${MAT_DIR}/rgb8.jpg")
endif()

# opencl
if(HAVE_OPENCL)
	extra_file(test_opencl.cl)
	tiny_test(test_opencl)
	extra_file(test_opencl_multi_array.cl)
	tiny_test(test_opencl_multi_array)
	tiny_test(test_opencl_fft)
endif()
