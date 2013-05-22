macro(tiny_test NAME)
	add_executable(${NAME} "${NAME}.cpp")
	target_link_libraries(${NAME} ${CMAKE_REQUIRED_LIBRARIES})
	add_test(
		NAME ${NAME}
		COMMAND ${NAME} ${ARGN}
	)
endmacro()

set(MAT_DIR "${CMAKE_SOURCE_DIR}/matfiles")

# Tests: return 0 for success.



# multi_array <-> uBLAS adaptors
#tiny_test(test_multi_array_matrix)
#tiny_test(test_multi_array_vector)

# multi_array tools
#tiny_test(test_multi_array_operators)
#tiny_test(test_multi_array_print)

# multi_array <-> FFTw adaptors
#tiny_test(test_multi_array_fft)

# operator split
#tiny_test(test_tuple_cut)
#tiny_test(test_operator_split)

# io
#if(HAVE_MATIO)
#	tiny_test(test_matlab_io "${MAT_DIR}/test.mat")
#endif()
#tiny_test(test_plain_io)

#if(HAVE_GIL)
#	tiny_test(test_gil_io "${MAT_DIR}/rgb8.jpg")
#endif()

# opencl
#if(HAVE_OPENCL)
#	tiny_test(test_opencl)
#	tiny_test(test_opencl_multi_array)
#endif()

#tiny_test(test_kernel_generator)
#tiny_test(test_resolvent)
tiny_test(test_constraint_parser)
tiny_test(test_resolvent_parser)
tiny_test(convolution)

tiny_test(convolution_error)
