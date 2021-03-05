#pragma once

#include <random>
#include <curand_kernel.h>

#include <koshi/OptixHelpers.h>
#include <koshi/Koshi.h>

#include "koshi/random/BlueNoiseRankingTiles_1.h"
#include "koshi/random/BlueNoiseScramblingTiles_1.h"
#include "koshi/random/BlueNoiseSobolSequence.h"

#define USE_BLUENOISE true

#define NUM_SOBOL_SAMPLES 256
#define INV_SOBOL_SAMPLES 1.f / 256.f
#define NUM_SOBOL_DIMENSIONS 256

#define NUM_OPTIMIZED_DIMENSIONS 8
#define BLUENOISE_WINDOW_SIZE 128


KOSHI_OPEN_NAMESPACE

class RandomGenerator;

struct RandomData
{
    Vec2u resolution;
    uint frame;
    int * d_scrambling_tiles;
    int * d_ranking_tiles;
    int * d_sobol_sequence;
};

class Random
{
public:
    DEVICE_FUNCTION float rand()
    {
#if USE_BLUENOISE
        int ranked_index = index ^ data->d_ranking_tiles[dimension + bluenoise_index];
        int value = data->d_sobol_sequence[dimension + ranked_index*NUM_SOBOL_DIMENSIONS];
        value = value ^ data->d_scrambling_tiles[(dimension%NUM_OPTIMIZED_DIMENSIONS) + bluenoise_index];
        dimension = (dimension + 1) % NUM_SOBOL_DIMENSIONS;
        return (curand_uniform(&state) + value) * INV_SOBOL_SAMPLES;
#else
        return curand_uniform(&state);
#endif
    }

private:
    DEVICE_FUNCTION Random(curandState_t& state, const Vec2u& pixel, const uint& _index, const RandomData * data)
    : index(_index % NUM_SOBOL_SAMPLES), dimension(0), state(state), data(data)
    {
        if(_index == 0)
            curand_init(pixel.x + pixel.y * data->resolution.x + data->frame * data->resolution.x * data->resolution.y, 0, 0, &state);
        bluenoise_index = (pixel.x % BLUENOISE_WINDOW_SIZE) + (pixel.y % BLUENOISE_WINDOW_SIZE)*BLUENOISE_WINDOW_SIZE;
        bluenoise_index *= NUM_OPTIMIZED_DIMENSIONS;
    }

    uint index;
    uint dimension;
    uint bluenoise_index;
    curandState_t& state;
    const RandomData * data;
    friend RandomGenerator;
};

class RandomGenerator
{
public:
    RandomGenerator() : states(nullptr) 
    {
        // TODO: Test if we actually need to copy this??? Can we just read directly from static const scrambling_tiles.
        CUDA_CHECK(cudaMalloc(&data.d_scrambling_tiles, sizeof(scrambling_tiles)));
        CUDA_CHECK(cudaMemcpy(data.d_scrambling_tiles, &scrambling_tiles, sizeof(scrambling_tiles), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&data.d_ranking_tiles, sizeof(ranking_tiles)));
        CUDA_CHECK(cudaMemcpy(data.d_ranking_tiles, &ranking_tiles, sizeof(ranking_tiles), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&data.d_sobol_sequence, sizeof(sobol_sequence)));
        CUDA_CHECK(cudaMemcpy(data.d_sobol_sequence, &sobol_sequence, sizeof(sobol_sequence), cudaMemcpyHostToDevice));
    }

    ~RandomGenerator()
    {
        CUDA_FREE(states);
        CUDA_FREE(data.d_scrambling_tiles);
        CUDA_FREE(data.d_ranking_tiles);
        CUDA_FREE(data.d_sobol_sequence);
    }

    void init(const Vec2u& resolution, const uint& frame)
    {
        if(!states || data.resolution != resolution)
        {
            CUDA_FREE(states);
            CUDA_CHECK(cudaMalloc(&states, sizeof(curandState_t) * resolution.x * resolution.y));
        }
        data.resolution = resolution;
        data.frame = frame;
    }

    DEVICE_FUNCTION Random get(const Vec2u& pixel, const uint& sample)
    {
        return Random(states[pixel.x + pixel.y * data.resolution.x], pixel, sample, &data);
    }

private:
    curandState_t * states;
    RandomData data;
};

KOSHI_CLOSE_NAMESPACE