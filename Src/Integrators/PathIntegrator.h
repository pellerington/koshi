#pragma once

#include "Integrator.h"

struct PathSample
{
    MaterialSample * msample = nullptr;
    LightSample    * lsample = nullptr;
    VolumeSample   * vsample = nullptr;

    enum Type { Camera, Material, Light, Volume };
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
    Vec3f integrate(Ray &ray) const
    {
        PathSample sample;
        sample.type = PathSample::Camera;
        sample.depth = 0;
        sample.quality = 1.f;
        return integrate(ray, sample);
    }
    Vec3f integrate(Ray &ray, PathSample &in_sample) const;
    Vec3f integrate_surface(const Intersect &intersect, PathSample &in_sample) const;
};
