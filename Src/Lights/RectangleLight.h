#pragma once

#include "Light.h"

class RectangleLight : public Light
{
public:
    RectangleLight(Vec3f position, Vec3f u, Vec3f v, Vec3f intensity, bool double_sided = false);
    bool evaluate_light(const Ray &ray, Vec3f &light, float* pdf = nullptr);
    bool sample_light(const uint num_samples, const Surface &surface, std::deque<SrfSample> &srf_samples);
    const uint estimated_samples(const Surface &surface) { return SAMPLES_PER_SA; }
private:
    Vec3f position;
    Vec3f u;
    Vec3f v;
    Vec3f normal;
    float area;
    bool double_sided;
};
