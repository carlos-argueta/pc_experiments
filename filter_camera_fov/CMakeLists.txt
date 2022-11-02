cmake_minimum_required(VERSION 2.6 FATAL_ERROR)


set(CMAKE_CXX_STANDARD 17)

project(filter)

find_package(PCL 1.10 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (filter filter.cpp)
target_link_libraries (filter ${PCL_LIBRARIES})
