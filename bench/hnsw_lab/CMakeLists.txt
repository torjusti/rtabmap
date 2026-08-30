cmake_minimum_required(VERSION 3.14)
project(hnsw_lab)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)

find_package(RTABMap REQUIRED COMPONENTS core)
# Match the same GTSAM build as librtabmap_core (TBB-enabled one).
set(GTSAM_DIR "$ENV{HOME}/.local/gtsam-tbb/lib/cmake/GTSAM" CACHE PATH "")
find_package(GTSAM REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(hnsw_lab main.cpp)
target_include_directories(hnsw_lab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib)
target_link_libraries(hnsw_lab rtabmap::core gtsam OpenMP::OpenMP_CXX)
