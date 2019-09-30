#pragma once

#include "Material.h"

class Lambert : public Material
{
public:
    Lambert(const Vec3f &diffuse_color = VEC3F_ZERO, const Vec3f &emission = VEC3F_ZERO);
    std::shared_ptr<Material> instance(const Surface &surface);
    const Vec3f get_emission() { return emission; }
    bool sample_material(const Surface &surface, std::deque<MaterialSample> &samples, float sample_reduction = 1.f);
    bool evaluate_material(const Surface &surface, MaterialSample &sample);

private:
    const Vec3f diffuse_color;
    const Vec3f emission;
};
