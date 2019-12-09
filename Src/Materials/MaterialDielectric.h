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
    std::shared_ptr<Material> instance(const Surface * surface);

    Type get_type() { return Material::Dielectric; }

    bool sample_material(std::vector<MaterialSample> &samples, float sample_reduction = 1.f);
    bool evaluate_material( MaterialSample &sample);
    const float get_ior() { return ior; }

private:
    std::shared_ptr<Fresnel> fresnel;
    const float ior;
    std::shared_ptr<MaterialGGXReflect> ggx_reflect;
    std::shared_ptr<MaterialGGXRefract> ggx_refract;
};
