#pragma once

#include "Material.h"
#include "Fresnel.h"
#include "GGX.h"

class MaterialGGXReflect : public Material
{
public:
    MaterialGGXReflect(const AttributeVec3f &specular_color_attribute, const AttributeFloat &roughness_attribute, std::shared_ptr<Fresnel> fresnel);
    std::shared_ptr<Material> instance(const Surface * surface);

    Type get_type() { return Material::GGXReflect; }

    bool sample_material(std::vector<MaterialSample> &samples, float sample_reduction = 1.f);
    bool evaluate_material(MaterialSample &sample);
    void set_fresnel(std::shared_ptr<Fresnel> _fresnel) { fresnel = _fresnel; }

private:
    const AttributeVec3f specular_color_attribute;
    Vec3f specular_color;

    const AttributeFloat roughness_attribute;
    float roughness;
    float roughness_sqr;

    std::shared_ptr<Fresnel> fresnel;
};
