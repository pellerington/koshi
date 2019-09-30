#pragma once

#include "Material.h"
#include "Fresnel.h"
#include "GGX.h"

class GGXRefract : public Material
{
public:
    GGXRefract(const Vec3f &refractive_color = VEC3F_ZERO, const float &roughness = 0.f, const float &ior = 1.f, std::shared_ptr<Fresnel> fresnel = nullptr, const Vec3f &emission = VEC3F_ZERO);
    std::shared_ptr<Material> instance(const Surface &surface);
    const Vec3f get_emission() { return emission; }
    bool sample_material(const Surface &surface, std::deque<MaterialSample> &samples, const float sample_reduction = 1.f);
    bool evaluate_material(const Surface &surface, MaterialSample &sample);
    void set_fresnel(std::shared_ptr<Fresnel> _fresnel) { fresnel = _fresnel; }

private:
    const Vec3f refractive_color;
    const float roughness;
    const float roughness_sqr;
    const float roughness_sqrt;
    const float ior;
    std::shared_ptr<Fresnel> fresnel;
    const Vec3f emission;
};
