cmake_minimum_required(VERSION 3.12)
project (Koshi)


find_library(OPENIMAGEIO NAMES openimageio OpenImageIO)
message(STATUS "OpenImageIO: ${OPENIMAGEIO_LIBRARY}")

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen: ${EIGEN3_INCLUDE_DIR}")

find_package(SFML 2 COMPONENTS graphics window system)

file(GLOB_RECURSE SOURCES "Src/*.cpp")
add_executable(koshi ${SOURCES})

set(CMAKE_CXX_FLAGS "-Wall -O3")
include_directories(${EIGEN3_INCLUDE_DIR})
target_link_libraries(koshi "-pthread")
target_link_libraries(koshi ${OPENIMAGEIO_LIBRARY})
target_link_libraries(koshi sfml-graphics sfml-window sfml-system)
