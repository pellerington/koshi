#pragma once

#include "Material.h"

class Lambert : public Material
{
public:
    Lambert(Vec3f diffuse_color = Vec3f::Zero(), Vec3f emission = Vec3f::Zero());
    const Vec3f get_emission() { return emission; }
    bool sample_material(const Surface &surface, std::deque<SrfSample> &srf_samples, float sample_reduction = 1.f);
    bool evaluate_material(const Surface &surface, SrfSample &srf_sample, float &pdf);
    
private:
    Vec3f diffuse_color;
    Vec3f emission;
};
