
# Stick common SMRE stuff into a library, used by tests etc.
set(BASE_SRC  matlab_io.cpp smre_data_double.cpp)

add_library(smre SHARED ${BASE_SRC})
target_link_libraries(smre ${CMAKE_REQUIRED_LIBRARIES})

set(CMAKE_REQUIRED_LIBRARIES smre ${CMAKE_REQUIRED_LIBRARIES})


# Executables using that library
add_executable(main  main.cpp)
target_link_libraries(main ${CMAKE_REQUIRED_LIBRARIES})
#add_dependencies(main smre)
