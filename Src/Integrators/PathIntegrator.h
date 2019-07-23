#pragma once

#include "Integrator.h"

class PathIntegrator : public Integrator
{
public:
    PathIntegrator(Scene * scene) : Integrator(scene) {}
    const Vec3f get_color();
    void pre_render();
    inline std::shared_ptr<Integrator> create(Ray &ray) { return create(ray, nullptr); }
    std::shared_ptr<Integrator> create(Ray &ray, float* light_pdf);
    void setup(Ray &ray, float* light_pdf);
    void integrate(size_t num_samples);
    const size_t get_required_samples() { return srf_samples.size(); }
    const bool completed() { return srf_samples.empty(); }
private:
    Surface surface;
    std::shared_ptr<Material> material;
    uint depth;
    Vec3f emission;
    Vec3f color;
    std::deque<SrfSample> srf_samples;
    float normalization;

    bool multiple_importance_sample;
    float light_sample_weight;
    float material_sample_weight;
};
