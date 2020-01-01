#pragma once

#include <vector>
#include <iostream>

#include "../Math/Types.h"
#include "../Scene/Scene.h"
#include "../Integrators/PathIntegrator.h"

struct Pixel
{
    Pixel(uint x, uint y, uint required_samples, std::vector<uint> &seeds, const RNG &rng)
    : pixel(x, y), color(VEC3F_ZERO), required_samples(required_samples), current_sample(0), seeds(std::move(seeds)), rng(std::move(rng))
    {}
        
    Vec2u pixel;
    Vec3f color;
    uint required_samples;
    uint current_sample;
    std::vector<uint> seeds;
    RNG rng;
};

class Render
{
public:
    Render(Scene * scene, const uint &num_workers);
    void start_render();
    void render_worker(const uint id, const std::vector<Vec2i> &work);
    Vec3f get_pixel_color(uint x, uint y) const;
    inline Vec2u get_image_resolution() const { return scene->camera.get_image_resolution(); }
    void kill_render() { kill_signal = true; }
private:
    std::unique_ptr<Integrator> integrator;
    const Scene * scene;
    const uint num_workers;
    const Vec2u resolution;
    std::vector<std::vector<Pixel>> pixels;
    bool kill_signal = false;
};
