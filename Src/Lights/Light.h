#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "../Math/Types.h"
#include "../Util/Surface.h"
#include "../Util/Ray.h"

#include <queue>
#include <iostream>

class Light
{
public:
    enum Type {
        None,
        Environment,
        Rectangle
    };
    Light(Type type, Vec3f intensity) : type(type), intensity(intensity) { }
    virtual bool evaluate_light(const Ray &ray, Vec3f &light, float* pdf = nullptr) = 0;
    virtual bool sample_light(const uint num_samples, const Surface &surface, std::deque<SrfSample> &srf_samples) = 0;
    virtual const uint estimated_samples(const Surface &surface) = 0;
    const Type type;
protected:
    Vec3f intensity;
};
