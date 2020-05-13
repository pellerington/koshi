#pragma once

#include <random>
#include <deque>
#include <algorithm>
#include <sstream>

#include <Math/Types.h>
#include <Math/BlueNoiseGenerator.h>

#define NUM_MAPS 2048
#define NUM_POINTS 128

class RandomNumberService;

class RandomNumberGen2D
{
public:
    RandomNumberGen2D(const uint& seed = 0) : i(seed % num_maps)
    {        
    }

    const Vec2f& rand() const
    {
        if(j >= num_points)
        {
            j = 0;
            i = (i + 1) % num_maps;
        }
        return maps[i][j++];
    }

private:
    mutable uint i, j;

    static Vec2f ** maps;
    static uint num_maps;
    static uint num_points;

    friend RandomNumberService;
};

class RandomNumberService
{
public:
    RandomNumberService(const uint& seed = 0) : random_generator(seed)
    {
    }

    void pre_render()
    {
        RandomNumberGen2D::num_maps = NUM_MAPS;
        RandomNumberGen2D::num_points = NUM_POINTS;
        RandomNumberGen2D::maps = BlueNoiseGenerator::GenerateMaps2D(RandomNumberGen2D::num_points, RandomNumberGen2D::num_maps);
    }

    RandomNumberGen2D get_random_2D()
    {
        return RandomNumberGen2D(random_generator());
    }

    template <class T>
    void shuffle(std::vector<T> &samples) { std::shuffle(samples.begin(), samples.end(), random_generator); }
    template <class T>
    void shuffle(std::deque<T> &samples) { std::shuffle(samples.begin(), samples.end(), random_generator); }

private:
    std::mt19937 random_generator;
};

// class RandomNumberGenerator1D