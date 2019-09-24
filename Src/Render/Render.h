#pragma once

#include <vector>
#include <iostream>

#include "../Math/Types.h"
#include "../Scene/Scene.h"
#include "../Integrators/PathIntegrator.h"

struct Pixel
{
    Vec2u pixel;
    Vec3f color = VEC3F_ZERO;
    uint required_samples = 0;
    float current_sample = 0.f;
    std::vector<Vec2f> rng;
};

class Render
{
public:
    Render(Scene * scene, const uint &num_workers);
    void start_render();
    void render_worker(const uint id, const std::vector<Vec2i> &work);
    Vec3f get_pixel_color(uint x, uint y) const;
    inline Vec2u get_image_resolution() const { return scene->camera.get_image_resolution(); }
private:
    std::unique_ptr<Integrator> integrator;
    const Scene * scene;
    const uint num_workers;
    const Vec2u resolution;
    std::vector<std::vector<Pixel>> pixels;
};
