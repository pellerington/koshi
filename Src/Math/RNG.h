#pragma once

#include <random>
#include <deque>
#include "../Math/Types.h"

class RNG
{
public:
    inline static float Rand() { return distribution(generator); };

    static void Rand2d(const uint &num_samples, std::vector<Vec2f> &samples)
    {
        samples.resize(num_samples);

        const float size = floor(sqrtf((float)num_samples));
        const float interval = 1.f / size;
        for(float u = 0; u < size; u++)
            for(float v = 0; v < size; v++)
                samples[u*size + v] = Vec2f(u*interval + Rand()*interval, v*interval + Rand()*interval);

        const uint size_sqr = size*size;
        uint remainder = num_samples - size_sqr;
        while(remainder--)
            samples[remainder + size_sqr] = Vec2f(Rand(), Rand());

        // static const float g = 1.32471795724474602596f;
        // static const float a1 = 1.f/g;
        // static const float a2 = 1.f/(g*g);
        //
        // for(uint i = 0; i < samples.size(); i++)
        // {
        //     samples[i] = Vec2f(fmod(0.5f+a1*n, 1), fmod(0.5f+a2*n, 1));
        //     n++;
        // }
    }

    template <class T>
    static void Shuffle(std::vector<T> &samples) { std::shuffle(samples.begin(), samples.end(), generator); }
    template <class T>
    static void Shuffle(std::deque<T> &samples) { std::shuffle(samples.begin(), samples.end(), generator); }

private:
    static std::mt19937 generator;
    static std::uniform_real_distribution<float> distribution;
    static double n;
};
