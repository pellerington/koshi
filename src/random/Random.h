#pragma once

#include <random>
#include <deque>
#include <algorithm>
#include <sstream>

#include <koshi/math/Types.h>

#define SEQUENCES 4096u
#define SEQUENCE_SIZE 128u

class RandomService;

template<uint N>
class Random
{
public:
    Random(const uint& seed = 0) : i(SEQUENCE_SIZE * N * (seed % SEQUENCES) - N)
    {
        // TODO: zero gets sampled a worrying amount for N == 2...
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
    RandomService(const uint& seed = 0) : random_generator(seed), distribution(0.f, 1.f)
    {
    }

    void pre_render();

    template<uint N>
    inline Random<N> get_random()
    {
        return Random<N>(random_generator());
    }

    inline float rand()
    {
        return distribution(random_generator);
    }

    inline uint seed()
    {
        return random_generator();
    }

private:
    std::mt19937 random_generator;
    std::uniform_real_distribution<float> distribution;
};