cmake_minimum_required(VERSION 3.12)
project (Koshi)

find_library(OPENIMAGEIO NAMES openimageio OpenImageIO)
message(STATUS "OpenImageIO: ${OPENIMAGEIO_LIBRARY}")

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen: ${EIGEN3_INCLUDE_DIR}")

find_package(embree 3.0 REQUIRED)
message(STATUS "embree include: ${EMBREE_INCLUDE_DIRS}")
message(STATUS "embree library: ${EMBREE_LIBRARY}")

find_package(SFML 2 COMPONENTS graphics window system)

file(GLOB_RECURSE SOURCES "Src/*.cpp")
add_executable(koshi ${SOURCES})

set(CMAKE_CXX_FLAGS "-Wall -O3")

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${EMBREE_INCLUDE_DIRS})

target_link_libraries(koshi ${EMBREE3_LIBRARY})
target_link_libraries(koshi /usr/lib64/libembree3.so.3)
target_link_libraries(koshi ${OPENIMAGEIO_LIBRARY})
target_link_libraries(koshi sfml-graphics sfml-window sfml-system)
target_link_libraries(koshi -pthread)
