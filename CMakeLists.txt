cmake_minimum_required(VERSION 3.12)
project (Koshi)

find_library(OPENIMAGEIO_LIBRARY NAMES openimageio OpenImageIO)
message(STATUS "OpenImageIO: ${OPENIMAGEIO_LIBRARY}")

find_package(embree 3.0 REQUIRED HINTS ~/embree-3.9.0.x86_64.linux/)
message(STATUS "embree include: ${EMBREE_INCLUDE_DIRS}")
message(STATUS "embree library: ${EMBREE_LIBRARIES}")

list(APPEND CMAKE_MODULE_PATH /usr/local/lib/cmake/OpenVDB/)
find_package(OpenVDB QUIET REQUIRED)
message(STATUS "openvdb include: ${OpenVDB_INCLUDE_DIRS}")
message(STATUS "openvdb library: ${OpenVDB_LIBRARIES}")

find_package(SFML 2 COMPONENTS graphics window system)

file(GLOB_RECURSE SOURCES "koshi/*.cc")

add_library(koshi SHARED ${SOURCES})
install(TARGETS koshi DESTINATION ${CMAKE_SOURCE_DIR}/build/lib)
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/koshi # source directory
    DESTINATION ${CMAKE_SOURCE_DIR}/build/include # target directory
    FILES_MATCHING PATTERN *.h
)

file(GLOB_RECURSE CMD_SOURCES "koshi-cmd/*.cc")
add_executable(koshi-cmd ${CMD_SOURCES})
add_dependencies(koshi-cmd koshi)

set(CMAKE_CXX_FLAGS "-Wall -msse4.1")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

message(${CMAKE_SOURCE_DIR}/koshi)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${EMBREE_INCLUDE_DIRS})
include_directories(${OpenVDB_INCLUDE_DIRS})

target_link_libraries(koshi ${EMBREE_LIBRARY})
target_link_libraries(koshi ${OpenVDB_LIBRARIES})
target_link_libraries(koshi ${OPENIMAGEIO_LIBRARY})
#target_link_libraries(koshi sfml-graphics sfml-window sfml-system)
target_link_libraries(koshi -pthread)
target_link_libraries(koshi -lHalf)
target_link_libraries(koshi -ltbb)

target_link_libraries(koshi-cmd ${CMAKE_SOURCE_DIR}/build/libkoshi.so)
target_link_libraries(koshi-cmd ${EMBREE_LIBRARY})
target_link_libraries(koshi-cmd ${OpenVDB_LIBRARIES})
target_link_libraries(koshi-cmd ${OPENIMAGEIO_LIBRARY})
target_link_libraries(koshi-cmd sfml-graphics sfml-window sfml-system)
target_link_libraries(koshi-cmd -pthread)
target_link_libraries(koshi-cmd -lHalf)
target_link_libraries(koshi-cmd -ltbb)
