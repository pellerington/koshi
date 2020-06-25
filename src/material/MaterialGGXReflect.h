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
    MaterialGGXReflect(const Texture * color_texture, const Texture * roughness_texture);
    MaterialInstance instance(const Surface * surface, const Intersect * intersect, Resources &resources);

private:
    const Texture * color_texture;
    const Texture * roughness_texture;
};
