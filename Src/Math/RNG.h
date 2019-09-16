#pragma once

#include <random>
#include <deque>
#include "../Math/Types.h"

class RNG
{
public:
    inline static float Rand() { return distribution(generator); };

    static void Rand2d(uint num_samples, std::vector<Vec2f> &samples)
    {
        float size = floor(sqrtf((float)num_samples));
        samples.resize(num_samples);
        float interval = 1.f / size;
        for(float u = 0; u < size; u++)
            for(float v = 0; v < size; v++)
                samples[u*size + v] = Vec2f(u*interval + Rand()*interval, v*interval + Rand()*interval);

        uint size_sqr = size*size;
        uint remainder = num_samples - size_sqr;
        while(remainder--)
            samples[remainder + size_sqr] = Vec2f(Rand(), Rand());
    }

    template <class T>
    static void Shuffle(std::vector<T> &samples) { std::shuffle(samples.begin(), samples.end(), generator); }
    template <class T>
    static void Shuffle(std::deque<T> &samples) { std::shuffle(samples.begin(), samples.end(), generator); }

private:
    static std::mt19937 generator;
    static std::uniform_real_distribution<float> distribution;
};
