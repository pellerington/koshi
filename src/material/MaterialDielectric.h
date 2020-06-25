#pragma once

#include <material/Material.h>
#include <material/Fresnel.h>

class MaterialDielectric : public Material
{
public:
    MaterialDielectric(const AttributeVec3f& reflective_color_attribute,
                       const AttributeVec3f& refractive_color_attribute,
                       const float& refractive_color_depth, 
                       const AttributeFloat& roughness_attribute,
                       const float& ior);

    MaterialInstance instance(const Surface * surface, const Intersect * intersect, Resources &resources);

private:
    const AttributeVec3f reflective_color_attribute;
    const AttributeVec3f refractive_color_attribute;
    const float refractive_color_depth;
    const AttributeFloat roughness_attribute;
    const float ior;
};
