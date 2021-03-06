#pragma once

#include <koshi/material/Material.h>
#include <koshi/material/Fresnel.h>
#include <koshi/material/GGX.h>

struct MaterialLobeGGXRefract : public MaterialLobe
{
    Fresnel * fresnel;
    float roughness_sqr;
    float ior_in;
    float ior_out;

    bool sample(MaterialSample& sample, Resources& resources) const;
    bool evaluate(MaterialSample& sample, Resources& resources) const;

    ScatterType get_scatter_type() const { return (roughness > EPSILON_F) ? ScatterType::GLOSSY : ScatterType::SPECULAR; }
    Hemisphere get_hemisphere() const { return Hemisphere::BACK; }
};

class MaterialGGXRefract : public Material
{
public:
    MaterialGGXRefract(const Texture * color_texture, const Texture * roughness_texture, 
                       const float& ior, const float& color_depth, 
                       const Texture * normal_texture, const Texture * opacity_texture);
    MaterialLobes instance(const Surface * surface, const Intersect * intersect, Resources& resources);

private:
    const Texture * color_texture;
    const float color_depth;
    const Texture * roughness_texture;
    const float ior;
};
