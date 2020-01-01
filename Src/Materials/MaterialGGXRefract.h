#pragma once

#include "Material.h"
#include "Fresnel.h"
#include "GGX.h"

class MaterialGGXRefract : public Material
{
public:
    MaterialGGXRefract(const AttributeVec3f &refractive_color_attribute, const AttributeFloat &roughness_attribute, const float &ior = 1.f, std::shared_ptr<Fresnel> fresnel = nullptr);
    std::shared_ptr<Material> instance(const Surface * surface, RNG &rng);

    Type get_type() { return Material::GGXRefract; }

    bool sample_material(std::vector<MaterialSample> &samples, const float sample_reduction = 1.f);
    bool evaluate_material(MaterialSample &sample);
    const float get_ior() { return ior; }
    void set_fresnel(std::shared_ptr<Fresnel> _fresnel) { fresnel = _fresnel; }

private:
    const AttributeVec3f refractive_color_attribute;
    Vec3f refractive_color;

    const AttributeFloat roughness_attribute;
    float roughness;
    float roughness_sqr;

    const float ior;
    float ior_in = 1.f, ior_out = 1.f;
    std::shared_ptr<Fresnel> fresnel;
};
