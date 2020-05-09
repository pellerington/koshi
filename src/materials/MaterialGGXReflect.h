#pragma once

#include <materials/Material.h>
#include <materials/Fresnel.h>
#include <materials/GGX.h>

struct MaterialLobeGGXReflect : public MaterialLobe
{
    Fresnel * fresnel;
    float roughness_sqr;

    bool sample(MaterialSample& sample, Resources& resources) const;
    Vec3f weight(const Vec3f& wo, Resources& resources) const;
    float pdf(const Vec3f& wo, Resources& resources) const;

    Type type() const { return (roughness > EPSILON_F) ? Type::Glossy : Type::Specular; }
};

class MaterialGGXReflect : public Material
{
public:
    MaterialGGXReflect(const AttributeVec3f &color_attribute, const AttributeFloat &roughness_attribute);
    MaterialInstance instance(const GeometrySurface * surface, Resources &resources);

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
