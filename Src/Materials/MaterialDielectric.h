#pragma once

#include "Material.h"
#include "Fresnel.h"
#include "MaterialGGXReflect.h"
#include "MaterialGGXRefract.h"

class MaterialDielectric : public Material
{
public:
    MaterialDielectric(const AttributeVec3f &reflective_color_attribute,
                       const AttributeVec3f &refractive_color_attribute,
                       const AttributeFloat &roughness_attribute,
                       const float &ior = 1.f);
    Type get_type() { return Material::Dielectric; }

    std::shared_ptr<MaterialInstance> instance(const Surface * surface);
    struct MaterialInstanceDielectric : public MaterialInstance
    {
        MaterialGGXReflect::MaterialInstanceGGXReflect reflect_instance;
        MaterialGGXRefract::MaterialInstanceGGXRefract refract_instance;
    };

    bool sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, RNG &rng, const float sample_reduction);
    bool evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample);

    const float get_ior() { return ior; }

private:
    const AttributeVec3f reflective_color_attribute;
    const AttributeVec3f refractive_color_attribute;
    const AttributeFloat roughness_attribute;
    const float ior;

    MaterialGGXReflect ggx_reflect;
    MaterialGGXRefract ggx_refract;
};
