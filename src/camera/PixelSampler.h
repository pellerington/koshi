#pragma once

#include <math/Types.h>
#include <vector>

// TODO: Figure out how to let users inject this into the camera.

typedef Vec2f (PixelSampleCallback)(const float rng[2]);

class BoxFilterSampler
{
public:
    static Vec2f sample(const float rng[2]) { return Vec2f(rng[0], rng[1]); }
};

class GaussianFilterSampler
{
public:
    static Vec2f sample(const float rng[2]) ;
private:
    static const std::vector<float> cdf;
};
