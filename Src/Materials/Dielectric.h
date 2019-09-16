#pragma once

#include "Material.h"
#include "Fresnel.h"
#include "GGXReflect.h"
#include "GGXRefract.h"

class Dielectric : public Material
{
public:
    Dielectric(Vec3f reflective_color = Vec3f::Zero(), Vec3f refractive_color = Vec3f::Zero(),
               float roughness = 0.f, float ior = 1.f, Vec3f emission = Vec3f::Zero());
    std::shared_ptr<Material> instance(const Surface &surface);
    const Vec3f get_emission() { return emission; }
    bool sample_material(const Surface &surface, std::deque<SrfSample> &srf_samples, float sample_reduction = 1.f);
    bool evaluate_material(const Surface &surface, SrfSample &srf_sample, float &pdf);

private:
    // Vec3f reflective_color;
    // Vec3f refractive_color;
    // float roughness;
    // float roughness_sqr;
    // float roughness_sqrt;
    std::shared_ptr<Fresnel> fresnel;
    float ior;
    Vec3f emission;
    std::shared_ptr<GGXReflect> ggx;
    std::shared_ptr<GGXRefract> ggx_refract;
};
