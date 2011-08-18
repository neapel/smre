
set(CMAKE_REQUIRED_LIBRARIES smre ${CMAKE_REQUIRED_LIBRARIES})


add_executable(test_smre_data_double test_smre_data_double.cpp)
target_link_libraries(test_smre_data_double ${CMAKE_REQUIRED_LIBRARIES})
add_test(
	NAME test_smre_data_double
	COMMAND test_smre_data_double
)
add_dependencies(test_smre_data_double smre)
