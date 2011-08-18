
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
endmacro(tiny_test)


# Tests: return 0 for success.
tiny_test(test_smre_data_double)
tiny_test(test_mat ../matfiles/test.mat)
