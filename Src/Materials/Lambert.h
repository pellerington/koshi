#pragma once

#include "Material.h"

class Lambert : public Material
{
public:
    Lambert(const Vec3f &diffuse_color = VEC3F_ZERO, const Vec3f &emission = VEC3F_ZERO);
    std::shared_ptr<Material> instance(const Surface &surface);
    const Vec3f get_emission() { return emission; }
    bool sample_material(const Surface &surface, std::deque<PathSample> &path_samples, float sample_reduction = 1.f);
    bool evaluate_material(const Surface &surface, PathSample &path_sample, float &pdf);

private:
    const Vec3f diffuse_color;
    const Vec3f emission;
};
