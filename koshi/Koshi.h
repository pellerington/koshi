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

typedef unsigned int uint;

#define KOSHI_DEBUG false