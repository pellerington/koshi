#pragma once

#include <materials/Material.h>
#include <materials/Fresnel.h>
#include <materials/GGX.h>

struct MaterialLobeGGXRefract : public MaterialLobe
{
    Fresnel * fresnel;
    float roughness_sqr;
    float ior_in;
    float ior_out;

    bool sample(MaterialSample& sample, Resources& resources) const;
    Vec3f weight(const Vec3f& wo, Resources& resources) const;
    float pdf(const Vec3f& wo, Resources& resources) const;

    Type type() const { return (roughness > EPSILON_F) ? Type::Glossy : Type::Specular; }
};

class MaterialGGXRefract : public Material
{
public:
    MaterialGGXRefract(const AttributeVec3f& color_attribute, const AttributeFloat& roughness_attribute, const float& ior, const float& color_depth = 0.f);
    MaterialInstance instance(const GeometrySurface * surface, Resources& resources);

private:
    const AttributeVec3f color_attribute;
    const float color_depth;
    const AttributeFloat roughness_attribute;
    const float ior;
};
