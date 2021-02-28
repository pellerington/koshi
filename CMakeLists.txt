cmake_minimum_required(VERSION 3.8)
project (HdKoshi)

# set(CMAKE_CXX_COMPILER nvcc)
# set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(pxr REQUIRED HINTS /usr/local/USD/)
message(STATUS "pxr include: ${PXR_INCLUDE_DIRS}")
message(STATUS "pxr library: ${PXR_LIBRARIES}")

find_package(CUDA REQUIRED)
message(STATUS "cuda include: ${CUDA_INCLUDE_DIRS}")
message(STATUS "cuda library: ${CUDA_LIBRARIES}")
enable_language(CUDA)

set(OptiX_INCLUDE "/home/peter/NVIDIA-OptiX-SDK-7.2.0-linux64-x86_64/include/")
message(STATUS "OptiX include: ${OptiX_INCLUDE}")

file(GLOB_RECURSE HD_KOSHI_SOURCES "src/hdKoshi/*.cpp" "src/koshi/*.cc")
add_library(HdKoshi SHARED ${HD_KOSHI_SOURCES})

file(GLOB_RECURSE PTX_KOSHI_SOURCES "src/koshi/*.cu")
add_library(PTXKoshi OBJECT ${PTX_KOSHI_SOURCES})
set_property(TARGET PTXKoshi PROPERTY CUDA_PTX_COMPILATION ON)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

#Common includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

#USD include
include_directories(${PXR_INCLUDE_DIRS})

#GPU includes
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OptiX_INCLUDE})

target_link_libraries(HdKoshi "-Wl,--disable-new-dtags")

#TODO: How do we properly import these from USD...
target_link_libraries(HdKoshi ${PXR_CMAKE_DIR}/lib/libIlmImf-2_2.so)
target_link_libraries(HdKoshi ${PXR_CMAKE_DIR}/lib/libOpenImageIO.so ${PXR_CMAKE_DIR}/lib/libIex-2_2.so ${PXR_CMAKE_DIR}/lib/libIlmImf-2_2.so)

target_link_libraries(HdKoshi ${PXR_LIBRARIES})
target_link_libraries(HdKoshi ${CUDA_LIBRARIES})