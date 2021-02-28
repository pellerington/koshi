#pragma once

#include <random>
#include <curand_kernel.h>

#include <koshi/OptixHelpers.h>
#include <koshi/Koshi.h>

#include "koshi/random/PMJ02.h"

#define NUM_1D_SAMPLES 8192u

KOSHI_OPEN_NAMESPACE

class RandomGenerator;

class Random
{
public:
    DEVICE_FUNCTION float rand()
    {
        return curand_uniform(&state);
    }

private:
    DEVICE_FUNCTION Random(curandState_t& state) : state(state) {}
	curandState_t& state;
    friend RandomGenerator;
};

class RandomGenerator
{
public:
    RandomGenerator() : states(nullptr) {}

    ~RandomGenerator()
    {
        CUDA_FREE(states);
    }

    void init(const Vec2u& _resolution, const uint& _frame)
    {
        // TODO: Instead of full resolution can we just use a 512x512 window and wrap around?
        if(!states || resolution != _resolution)
        {
            CUDA_FREE(states);
            CUDA_CHECK(cudaMalloc(&states, sizeof(curandState_t) * _resolution.x * _resolution.y));
        }
        resolution = _resolution;
        frame = _frame;
    }

    DEVICE_FUNCTION void init(const Vec2u& pixel)
    {
        const uint seed = pixel.x + pixel.y * resolution.x + frame * resolution.x * resolution.y;
        curand_init(seed, 0, 0, &states[pixel.x + pixel.y * resolution.x]);
    }

    DEVICE_FUNCTION Random get(const Vec2u& pixel)
    {
        return Random(states[pixel.x + pixel.y * resolution.x]);
    }

private:
    Vec2u resolution;
    uint frame;
    curandState_t * states;
};

KOSHI_CLOSE_NAMESPACE