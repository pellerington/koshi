#pragma once

#include <iostream>

#define CUDA_CHECK(call) \
{\
    cudaError_t rc = call;\
    if (rc != cudaSuccess) {\
        cudaError_t err = rc; /*cudaGetLastError();*/\
        std::cout << "CUDA Error " << cudaGetErrorName(err) << " (" << cudaGetErrorString(err) << ") " << __FILE__ << " " << __LINE__ << std::endl;\
    }\
}

#define OPTIX_CHECK(call)\
{\
    OptixResult res = call;\
    if(res != OPTIX_SUCCESS) {\
        fprintf(stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__);\
        exit(2);\
    }\
}

#define OPTIX_CHECK_LOG(call)\
{\
    OptixResult res = call;\
    if(res != OPTIX_SUCCESS) {\
        fprintf(stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__);\
        exit(2);\
    }\
    std::cout << log << "\n";\
}

#define CUDA_SYNC_CHECK()\
{\
    cudaDeviceSynchronize();\
    cudaError_t error = cudaGetLastError();\
    if(error != cudaSuccess) {\
        fprintf(stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error));\
        exit(2);\
    }\
}

#define CUDA_FREE(var)\
{\
    if(var != 0) {\
        cudaError_t rc = cudaFree(reinterpret_cast<void*>(var));\
        var = 0;\
        if (rc != cudaSuccess) {\
            cudaError_t err = rc; /*cudaGetLastError();*/\
            std::cout << "CUDA Error " << cudaGetErrorName(err) << " (" << cudaGetErrorString(err) << ") " << __FILE__ << " " << __LINE__ << std::endl;\
        }\
    }\
}

