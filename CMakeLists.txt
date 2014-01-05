cmake_minimum_required(VERSION 2.8)
project(homework_2)

add_definitions("--std=c++11")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${homework_2_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(PCL REQUIRED)

add_definitions(${PCL_DEFINITIONS})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

add_executable (ex1 src/ex1.cpp)
target_link_libraries(ex1 ${PCL_LIBRARIES})

add_executable (ex2 src/ex2.cpp)
target_link_libraries(ex2 ${PCL_LIBRARIES})

add_executable (ex3 src/ex3.cpp)
target_link_libraries(ex3 ${PCL_LIBRARIES})

add_executable (ex4 src/ex4.cpp)
target_link_libraries(ex4 ${PCL_LIBRARIES})

add_executable (ex5 src/ex5.cpp)
target_link_libraries(ex5 ${PCL_LIBRARIES})

add_executable (ex6 src/ex6.cpp)
target_link_libraries(ex6 ${PCL_LIBRARIES})
