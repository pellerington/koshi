#pragma once

#include "Integrator.h"

class PathIntegrator : public Integrator
{
public:
    PathIntegrator(Scene * scene) : Integrator(scene) {}
    void pre_render();
    Vec3f integrate(Ray &ray) const { return integrate(ray, 1.f, nullptr, nullptr); }
    Vec3f integrate(Ray &ray, const float current_quality, float* light_pdf, const LightSample* light_sample) const;
};
