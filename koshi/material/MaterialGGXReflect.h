#pragma once

#include <koshi/material/Material.h>
#include <koshi/material/Fresnel.h>
#include <koshi/material/GGX.h>

struct MaterialLobeGGXReflect : public MaterialLobe
{
    Fresnel * fresnel;
    float roughness_sqr;

    bool sample(MaterialSample& sample, Resources& resources) const;
    bool evaluate(MaterialSample& sample, Resources& resources) const;

    ScatterType get_scatter_type() const { return (roughness > EPSILON_F) ? ScatterType::GLOSSY : ScatterType::SPECULAR; }
    Hemisphere get_hemisphere() const { return Hemisphere::FRONT; }
};

class MaterialGGXReflect : public Material
{
public:
    MaterialGGXReflect(const Texture * color_texture, const Texture * roughness_texture, 
                       const Texture * normal_texture, const Texture * opacity_texture);
    MaterialLobes instance(const Surface * surface, const Intersect * intersect, Resources& resources);

private:
    const Texture * color_texture;
    const Texture * roughness_texture;
};
