# Stick common SMRE stuff into a library, used by tests etc.
set(BASE_SRC
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

add_executable(benchmark benchmark.cpp ${BASE_SRC})
target_link_libraries(benchmark ${CMAKE_REQUIRED_LIBRARIES})

add_executable(convolution_benchmark convolution_benchmark.cpp ${BASE_SRC})
target_link_libraries(convolution_benchmark ${CMAKE_REQUIRED_LIBRARIES})
