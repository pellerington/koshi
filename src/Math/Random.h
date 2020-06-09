#pragma once

#include <random>
#include <deque>
#include <algorithm>
#include <sstream>

#include <Math/Types.h>
#include <Math/BlueNoiseGenerator.h>

#define NUM_MAPS 2048
#define NUM_POINTS 128

class RandomService;

class Random2D
{
public:
    Random2D(const uint& seed = 0) : i(seed % num_maps)
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

    friend RandomService;
};

class RandomService
{
public:
    RandomService(const uint& seed = 0) : random_generator(seed)
    {
    }

    void pre_render()
    {
        Random2D::num_maps = NUM_MAPS;
        Random2D::num_points = NUM_POINTS;
        Random2D::maps = BlueNoiseGenerator::GenerateMaps2D(Random2D::num_points, Random2D::num_maps);
    }

    Random2D get_random_2D()
    {
        return Random2D(random_generator());
    }

    template <class T>
    void shuffle(std::vector<T> &samples) { std::shuffle(samples.begin(), samples.end(), random_generator); }
    template <class T>
    void shuffle(std::deque<T> &samples) { std::shuffle(samples.begin(), samples.end(), random_generator); }

private:
    std::mt19937 random_generator;
};

// class Random1D or Random<2> Random<1> ect.