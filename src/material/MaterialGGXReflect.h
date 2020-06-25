#pragma once

#include <material/Material.h>
#include <material/Fresnel.h>
#include <material/GGX.h>

struct MaterialLobeGGXReflect : public MaterialLobe
{
    Fresnel * fresnel;
    float roughness_sqr;

    bool sample(MaterialSample& sample, Resources& resources) const;
    Vec3f weight(const Vec3f& wo, Resources& resources) const;
    float pdf(const Vec3f& wo, Resources& resources) const;

    ScatterType get_scatter_type() const { return (roughness > EPSILON_F) ? ScatterType::GLOSSY : ScatterType::SPECULAR; }
    Hemisphere get_hemisphere() const { return Hemisphere::FRONT; }
};

class MaterialGGXReflect : public Material
{
public:
    MaterialGGXReflect(const AttributeVec3f &color_attribute, const AttributeFloat &roughness_attribute);
    MaterialInstance instance(const Surface * surface, const Intersect * intersect, Resources &resources);

private:
    const AttributeVec3f color_attribute;
    const AttributeFloat roughness_attribute;

    /* DEALING WITH FRESNEL:
        - include fresnel in the constructor.
        - if we didn't include one then default = FresnelMetallic/FrenelDielectric
        - if we include one and its nullptr then dont calculate it.

        - include a fresnel float in the Instance struct.
        - if we are computing our own frenel then add that here.
        - otherwise if its nullptr put 1.f.

        - fresnel object will need to change though.
    */
};
