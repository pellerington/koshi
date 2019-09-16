#pragma once

#include "Material.h"
#include "Fresnel.h"
#include "GGX.h"

class GGXRefract : public Material
{
public:
    GGXRefract(Vec3f refractive_color = Vec3f::Zero(), float roughness = 0.f, float ior = 1.f, std::shared_ptr<Fresnel> fresnel = nullptr, Vec3f emission = Vec3f::Zero());
    std::shared_ptr<Material> instance(const Surface &surface);
    const Vec3f get_emission() { return emission; }
    bool sample_material(const Surface &surface, std::deque<SrfSample> &srf_samples, float sample_reduction = 1.f);
    bool evaluate_material(const Surface &surface, SrfSample &srf_sample, float &pdf);
    void set_fresnel(std::shared_ptr<Fresnel> _fresnel) { fresnel = _fresnel; }

private:
    Vec3f refractive_color; // Make this const once we stopped using eigen.
    const float roughness;
    const float roughness_sqr;
    const float roughness_sqrt;
    const float ior;
    std::shared_ptr<Fresnel> fresnel;
    const Vec3f emission;
};
