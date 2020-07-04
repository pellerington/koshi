#pragma once

#include <random>
#include <deque>
#include <algorithm>
#include <sstream>

#include <math/Types.h>
#include <math/BlueNoiseGenerator.h>

#define SEQUENCES 2048u
#define SEQUENCE_SIZE 128u

class RandomService;

template<uint N>
class Random
{
public:
    Random(const uint& seed = 0) : i(SEQUENCE_SIZE * N * (seed % SEQUENCES) - N)
    {
    }

    inline const float * rand() const
    {
        i += N;
        if(i >= SEQUENCES * SEQUENCE_SIZE * N)
            i = 0;
        return &data[i];
    }

private:
    mutable uint i;
    static float * data;

    friend RandomService;
};

template class Random<1>;
template class Random<2>;

class RandomService
{
public:
    RandomService(const uint& seed = 0) : random_generator(seed)
    {
    }

    void pre_render();

    template<uint N>
    inline Random<N> get_random()
    {
        return Random<N>(random_generator());
    }

private:
    std::mt19937 random_generator;
};