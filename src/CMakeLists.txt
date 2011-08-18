
set(BASE_SRC  matlab_io.cpp smre_data_double.cpp)


add_executable(main  main.cpp ${BASE_SRC})
target_link_libraries(main ${CMAKE_REQUIRED_LIBRARIES})
