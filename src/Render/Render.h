#pragma once

#include <vector>
#include <iostream>

#include <Math/Types.h>
#include <base/Scene.h>
#include <intersection/Intersector.h>
#include <base/Settings.h>
#include <Math/Random.h>

struct Pixel
{
    Pixel(const uint x, const uint y, const uint required_samples, const uint seed, const Random2D& rng)
    : pixel(x, y), color(VEC3F_ZERO), required_samples(required_samples), current_sample(0), seed(seed), rng(rng)
    {
    }

    Vec2u pixel;
    Vec3f color;
    uint required_samples;
    uint current_sample;
    std::mt19937 seed;
    Random2D rng;
};

class Render
{
public:
    Render(Scene& scene, Settings& settings);
    void start_render();
    void render_worker(const uint id, const std::vector<Vec2i>& work);
    Vec3f get_pixel_color(const uint& x, const uint& y) const;
    inline Vec2u get_image_resolution() const { return resolution; }
    void kill_render() { kill_signal = true; }

private:
    Intersector * intersector;
    Scene& scene;
    const Camera * camera;
    Settings& settings;

    const Vec2u resolution;
    Pixel *** pixels; // <- Needs to be freed when renderer is killed.
    bool kill_signal = false;
};
