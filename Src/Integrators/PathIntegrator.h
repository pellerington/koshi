#pragma once

#include "Integrator.h"

struct PathSample
{
    MaterialSample * msample = nullptr;
    LightSample    * lsample = nullptr;

    enum Type { Material, Light };
    Type type = Type::Material;

    // PathSample * parent = nullptr;

    //Volumes which we are inside?
};

class PathIntegrator : public Integrator
{
public:
    PathIntegrator(Scene * scene) : Integrator(scene) {}
    void pre_render();
    Vec3f integrate(Ray &ray) const { return integrate(ray, 1.f, nullptr); }
    Vec3f integrate(Ray &ray, const float current_quality, PathSample * in_sample) const;
};
