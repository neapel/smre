add_executable(guimain guimain.cpp)
target_link_libraries(guimain ${CMAKE_REQUIRED_LIBRARIES})

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark ${CMAKE_REQUIRED_LIBRARIES})

add_executable(convolution_benchmark convolution_benchmark.cpp)
target_link_libraries(convolution_benchmark ${CMAKE_REQUIRED_LIBRARIES})
