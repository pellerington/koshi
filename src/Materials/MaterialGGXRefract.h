#pragma once

#include <Materials/Material.h>
#include <Materials/Fresnel.h>
#include <Materials/GGX.h>

class MaterialGGXRefract : public Material
{
public:
    MaterialGGXRefract(const AttributeVec3f &refractive_color_attribute, const AttributeFloat &roughness_attribute, const float &ior = 1.f);

    MaterialInstance * instance(const GeometrySurface * surface, Resources &resources);
    struct MaterialInstanceGGXRefract : public MaterialInstance
    {
        Vec3f refractive_color;
        Fresnel * fresnel;
        float roughness;
        float roughness_sqr;
        float ior_in;
        float ior_out;
    };

    bool sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, const float sample_reduction, Resources &resources);
    bool evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample);

    const float get_ior() { return ior; }

private:
    const AttributeVec3f refractive_color_attribute;
    const AttributeFloat roughness_attribute;
    const float ior;
};
