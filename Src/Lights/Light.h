#pragma once

#include "../Util/Surface.h"
#include "../Util/Ray.h"

#include <queue>
#include <iostream>

struct LightSample {
    Vec3f position;
    Vec3f intensity;
    float pdf = 0.f;
};

class Light
{
public:
    enum Type {
        None,
        Environment,
        Rectangle
    };
    Light(const Type &type, const Vec3f &intensity) : type(type), intensity(intensity) { }
    virtual bool evaluate_light(const Ray &ray, LightSample &light_sample) = 0;
    virtual bool sample_light(const uint num_samples, const Surface &surface, std::vector<LightSample> &light_samples) = 0;
    virtual const uint estimated_samples(const Surface &surface) = 0;
    const Type type;

protected:
    Vec3f intensity;
};
