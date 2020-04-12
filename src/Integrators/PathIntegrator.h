#pragma once

#include <Integrators/Integrator.h>

struct PathSample
{
    MaterialSample * msample = nullptr;
    LightSample    * lsample = nullptr;

    enum Type { Camera, Reflection, Transmission, Volume, Light };
    Type type;

    PathSample * parent;
    uint depth;
    float quality;
};

class PathIntegrator : public Integrator
{
public:
    PathIntegrator(Scene * scene) : Integrator(scene) {}
    void pre_render();
    Vec3f integrate(Ray &ray, Resources &resources) const
    {
        PathSample sample;
        sample.type = PathSample::Camera;
        sample.depth = 0;
        sample.quality = 1.f;
        LightSample lsample;
        sample.lsample = &lsample;
        return integrate(ray, sample, resources);
    }
    Vec3f integrate(Ray &ray, PathSample &in_sample, Resources &resources) const;
private:
    Vec3f scatter_surface(const Intersect &intersect, PathSample &in_sample, Resources &resources) const;
    // Scatter volume?
    double quality_threshold = 1.f;
};
