 #pragma once

#include <random>

#include <koshi/OptixHelpers.h>
#include <koshi/Koshi.h>

#define UNIFORM_RANDOM_SIZE 2048u

KOSHI_OPEN_NAMESPACE

class Random
{
public:
    Random() : d_seeds(nullptr)
    {
        std::mt19937 random_generator(INT32_MAX);
        std::uniform_real_distribution<float> distribution;

        float data[UNIFORM_RANDOM_SIZE];
        for(uint i = 0; i < UNIFORM_RANDOM_SIZE; i++)
            data[i] = distribution(random_generator);

        CUDA_CHECK(cudaMalloc(&d_data, UNIFORM_RANDOM_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_data), data, UNIFORM_RANDOM_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    }

    ~Random()
    {
        CUDA_CHECK(cudaFree(d_data));
        if(d_seeds != nullptr)
            CUDA_CHECK(cudaFree(d_seeds));
    }

    void init(const Vec2u& _resolution)
    {
        if(resolution != _resolution)
        {
            if(d_seeds != nullptr)
                CUDA_CHECK(cudaFree(d_seeds));
            CUDA_CHECK(cudaMalloc(&d_seeds, _resolution.x * _resolution.y * sizeof(uint)));
        }

        resolution = _resolution;
    }

    void setSeeds(const uint& sample, const uint& frame)
    {
        std::vector<uint> seeds(resolution.x*resolution.y);

        std::mt19937 random_generator(sample + frame * 512u);
        for(uint i = 0; i < resolution.x*resolution.y; i++)
            seeds[i] = random_generator();

        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_seeds), seeds.data(), resolution.x * resolution.y * sizeof(uint), cudaMemcpyHostToDevice));
    }

#if CUDA_COMPILE
    DEVICE_FUNCTION float rand()
    {
        const uint3 idx = optixGetLaunchIndex();
        return d_data[d_seeds[idx.x + idx.y*resolution.x]++ % UNIFORM_RANDOM_SIZE];
    }
#endif

private:
    float * d_data;

    Vec2u resolution;
    uint * d_seeds;
};

KOSHI_CLOSE_NAMESPACE