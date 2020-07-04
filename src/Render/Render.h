#pragma once

#include <vector>
#include <iostream>

#include <math/Types.h>
#include <base/Scene.h>
#include <intersection/Intersector.h>
#include <base/Settings.h>
#include <math/Random.h>
#include <camera/Camera.h>

struct Pixel
{
    Pixel(const uint x, const uint y, const uint seed, const Random<2>& rng)
    : color(VEC3F_ZERO), color_sqr(VEC3F_ZERO), samples(0.f), variance(0.f), seed(seed), rng(rng)
    {
    }

    Vec3f color;
    Vec3f color_sqr;    
    float samples;
    float variance;

    std::mt19937 seed;
    Random<2> rng;
};

class Render
{
public:
    Render(Scene& scene, Settings& settings);
    void start_render();
    void render_worker(const uint& id);
    Vec3f get_pixel_color(const uint& x, const uint& y) const;
    inline Vec2u get_image_resolution() const { return resolution; }
    void kill() { kill_render = true; }

private:
    Intersector * intersector;
    Scene& scene;
    const Camera * camera;
    Settings& settings;

    const Vec2u resolution;
    Pixel *** pixels; // <- Needs to be freed when renderer is killed.
    bool kill_render = false;

    std::vector<uint> pass_resolution;
    bool preview;
};
