#pragma once

#include "Material.h"
#include "Fresnel.h"
#include "GGX.h"

class MaterialGGXRefract : public Material
{
public:
    MaterialGGXRefract(const AttributeVec3f &refractive_color_attribute, const AttributeFloat &roughness_attribute, const float &ior = 1.f);
    Type get_type() { return Material::GGXRefract; }

    std::shared_ptr<MaterialInstance> instance(const Surface * surface);
    struct MaterialInstanceGGXRefract : public MaterialInstance
    {
        Vec3f refractive_color;
        std::shared_ptr<Fresnel> fresnel;
        float roughness;
        float roughness_sqr;
        float ior_in;
        float ior_out;
    };

    bool sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, RNG &rng, const float sample_reduction);
    bool evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample);

    const float get_ior() { return ior; }

private:
    const AttributeVec3f refractive_color_attribute;
    const AttributeFloat roughness_attribute;
    const float ior;
};
