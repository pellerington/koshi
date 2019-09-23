#pragma once

#include "Light.h"

class RectangleLight : public Light
{
public:
    RectangleLight(const Vec3f &position, const Vec3f &u, const Vec3f &v, const Vec3f &intensity, const bool double_sided = false);
    bool evaluate_light(const Ray &ray, Vec3f &light, float * pdf = nullptr);
    bool sample_light(const uint num_samples, const Surface &surface, std::deque<LightSample> &light_samples);
    const uint estimated_samples(const Surface &surface) { return SAMPLES_PER_SA; }
private:
    const Vec3f position;
    const Vec3f u;
    const Vec3f v;
    const Vec3f normal;
    const float area;
    const bool double_sided;
};
