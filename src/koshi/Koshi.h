#pragma once

#define KOSHI_OPEN_NAMESPACE namespace Koshi {
#define KOSHI_CLOSE_NAMESPACE }

#define TARGET GPU

#ifdef __CUDACC__
#define CUDA_COMPILE true
#endif

#ifdef CUDA_COMPILE
#define DEVICE_FUNCTION __forceinline__ __device__ 
#else
#define DEVICE_FUNCTION inline 
#endif

#include <cstdint>
typedef uint32_t uint;

#define KOSHI_DEBUG false

#define LOG_ERROR(error) std::cout << "KOSHI ERROR: " << error << std::endl;
