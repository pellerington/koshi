#pragma once

#include <materials/Material.h>
#include <materials/Fresnel.h>

class MaterialDielectric : public Material
{
public:
    MaterialDielectric(const AttributeVec3f &reflective_color_attribute,
                       const AttributeVec3f &refractive_color_attribute,
                       const AttributeFloat &roughness_attribute,
                       const float &ior = 1.f);

    MaterialInstance instance(const GeometrySurface * surface, Resources &resources);

private:
    const AttributeVec3f reflective_color_attribute;
    const AttributeVec3f refractive_color_attribute;
    const AttributeFloat roughness_attribute;
    const float ior;
};