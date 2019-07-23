#pragma once

#include <vector>
#include <iostream>

#include "../Math/Types.h"
#include "../Scene/Scene.h"
#include "../Integrators/PathIntegrator.h"

class Render
{
public:
    Render(Scene * scene, uint num_workers);
    void start_render();
    void render_worker(const uint id, const std::vector<Vec2i> &work);
    Vec3f get_pixel(uint x, uint y);
    Vec2i get_image_resolution() { return scene->camera.get_image_resolution(); }
private:
    struct PixelCtx
    {
        Vec2i pixel;
        Vec3f color = Vec3f::Zero();
        uint required_samples = 0;
        float current_sample = 0.f;
        std::vector<Vec2f> rng;
    };
    std::unique_ptr<Integrator> master_integrator;
    Scene * scene;
    std::vector<std::vector<PixelCtx>> pixels;
    uint num_workers;
};
