cmake_minimum_required(VERSION 3.14)
project(bench_export)

set(CMAKE_BUILD_TYPE Release)

find_package(RTABMap REQUIRED COMPONENTS core gui)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(bench_export main.cpp)
target_link_libraries(bench_export rtabmap::core OpenMP::OpenMP_CXX)

add_executable(gui_test gui_test.cpp)
target_link_libraries(gui_test rtabmap::gui OpenMP::OpenMP_CXX)
