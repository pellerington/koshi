cmake_minimum_required(VERSION 3.12)
project (Koshi)

list(APPEND CMAKE_MODULE_PATH /usr/local/lib/cmake/OpenVDB/)

find_library(OPENIMAGEIO_LIBRARY NAMES openimageio OpenImageIO)
message(STATUS "OpenImageIO: ${OPENIMAGEIO_LIBRARY}")

find_package(embree 3.0 REQUIRED HINTS ~/embree-3.9.0.x86_64.linux/)
message(STATUS "embree include: ${EMBREE_INCLUDE_DIRS}")
message(STATUS "embree library: ${EMBREE_LIBRARIES}")

find_package(OpenVDB QUIET REQUIRED)
message(STATUS "openvdb include: ${OpenVDB_INCLUDE_DIRS}")
message(STATUS "openvdb library: ${OpenVDB_LIBRARIES}")

find_package(SFML 2 COMPONENTS graphics window system)

file(GLOB_RECURSE SOURCES "src/*.cc")
add_executable(koshi ${SOURCES})

set(CMAKE_CXX_FLAGS "-Wall -msse4.1 -O3")
#set(CMAKE_CXX_FLAGS "-Wall -msse4.1 -g")

include_directories(${EMBREE_INCLUDE_DIRS})
include_directories(${OpenVDB_INCLUDE_DIRS})
include_directories(dependencies)	
include_directories(src)

#target_link_libraries(koshi /usr/lib64/libembree3.so.3)
target_link_libraries(koshi ${EMBREE_LIBRARY})
target_link_libraries(koshi ${OpenVDB_LIBRARIES})
target_link_libraries(koshi ${OPENIMAGEIO_LIBRARY})
target_link_libraries(koshi sfml-graphics sfml-window sfml-system)
target_link_libraries(koshi -pthread)
target_link_libraries(koshi -lHalf)
target_link_libraries(koshi -ltbb)
