#pragma once

#include "Material.h"
#include "Fresnel.h"
#include "GGX.h"

class MaterialGGXRefract : public Material
{
public:
    MaterialGGXRefract(const Vec3f &refractive_color = VEC3F_ZERO, const float &roughness = 0.f, const float &ior = 1.f, std::shared_ptr<Fresnel> fresnel = nullptr);
    std::shared_ptr<Material> instance(const Surface * surface);

    Type get_type() { return Material::GGXRefract; }

    bool sample_material(std::vector<MaterialSample> &samples, const float sample_reduction = 1.f);
    bool evaluate_material(MaterialSample &sample);
    const float get_ior() { return ior; }
    void set_fresnel(std::shared_ptr<Fresnel> _fresnel) { fresnel = _fresnel; }

private:
    const Vec3f refractive_color;
    const float roughness;
    const float roughness_sqr;
    const float roughness_sqrt;
    const float ior;
    std::shared_ptr<Fresnel> fresnel;
    float ior_in = 1.f, ior_out = 1.f;
};
