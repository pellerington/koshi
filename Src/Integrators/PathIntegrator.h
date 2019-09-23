#pragma once

#include "Integrator.h"

class PathIntegrator : public Integrator
{
public:
    PathIntegrator(Scene * scene) : Integrator(scene) {}
    Vec3f get_color();
    void pre_render();
    inline std::shared_ptr<Integrator> create(Ray &ray) { return create(ray, 1.f, nullptr, nullptr); }
    std::shared_ptr<Integrator> create(Ray &ray, const float current_quality, float* light_pdf, const LightSample* light_sample);
    void setup(Ray &ray, const float current_quality, float* light_pdf, const LightSample* light_sample);
    void integrate(const size_t num_samples);
    size_t get_required_samples() { return path_samples.size(); }
    bool completed() { return path_samples.empty(); }
private:
    Surface surface;
    std::shared_ptr<Material> material;
    uint depth;

    //Emission + Color seems redundant;
    Vec3f emission;
    Vec3f color;

    std::deque<PathSample> path_samples;
    float normalization;
    float current_quality;

    bool multiple_importance_sample;
    float light_sample_weight;
    float material_sample_weight;
};
