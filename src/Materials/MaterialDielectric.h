#pragma once

#include <Materials/Material.h>
#include <Materials/Fresnel.h>
#include <Materials/MaterialGGXReflect.h>
#include <Materials/MaterialGGXRefract.h>

class MaterialDielectric : public Material
{
public:
    MaterialDielectric(const AttributeVec3f &reflective_color_attribute,
                       const AttributeVec3f &refractive_color_attribute,
                       const AttributeFloat &roughness_attribute,
                       const float &ior = 1.f);
    Type get_type() { return Material::Dielectric; }

    MaterialInstance * instance(const Surface * surface, Resources &resources);
    struct MaterialInstanceDielectric : public MaterialInstance
    {
        MaterialGGXReflect::MaterialInstanceGGXReflect reflect_instance;
        MaterialGGXRefract::MaterialInstanceGGXRefract refract_instance;
    };

    bool sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, const float sample_reduction, Resources &resources);
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
