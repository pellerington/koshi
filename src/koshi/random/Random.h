#pragma once

#include <random>
#include <curand_kernel.h>

#include <koshi/OptixHelpers.h>
#include <koshi/Koshi.h>

#include "koshi/random/BlueNoiseRankingTiles_1.h"
#include "koshi/random/BlueNoiseScramblingTiles_1.h"
#include "koshi/random/BlueNoiseSobolSequence.h"

#define USE_BLUE_NOISE true

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
#if USE_BLUE_NOISE
        int ranked_index = index ^ data->d_ranking_tiles[dimension + (pixel.x + pixel.y*128)*8];
        int value = data->d_sobol_sequence[dimension + ranked_index*256];
        value = value ^ data->d_scrambling_tiles[dimension + (pixel.x + pixel.y*128)*8];
        dimension = (dimension + 1) % 256;
        return (curand_uniform(&state) + value) / 256.f; // TODO: DEFINE SOME OF THESE VALUES (128 / 256 ect).
#else
        return curand_uniform(&state);
#endif
    }

private:
    DEVICE_FUNCTION Random(curandState_t& state, const Vec2u& _pixel, const uint& _index, const RandomData * data)
    : index(_index % 256), dimension(0), pixel(Vec2u(_pixel.x % 128, _pixel.y % 128)), state(state), data(data)
    {
        if(_index == 0)
            curand_init(_pixel.x + _pixel.y * data->resolution.x + data->frame * data->resolution.x * data->resolution.y, 0, 0, &state);
    }

    uint index;
    uint dimension;
    Vec2u pixel;
    curandState_t& state;
    const RandomData * data;
    friend RandomGenerator;
};

class RandomGenerator
{
public:
    RandomGenerator() : states(nullptr) 
    {
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