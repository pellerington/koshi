#pragma once

#include <koshi/material/Material.h>
#include <koshi/material/Fresnel.h>

class MaterialDielectric : public Material
{
public:
    MaterialDielectric(const Texture * reflective_color_texture,
                       const Texture * refractive_color_texture,
                       const float& refractive_color_depth, 
                       const Texture * roughness_texture,
                       const float& ior,
                       const Texture * normal_texture, 
                       const Texture * opacity_texture);

    MaterialLobes instance(const Surface * surface, const Intersect * intersect, Resources& resources);

private:
    const Texture * reflective_color_texture;
    const Texture * refractive_color_texture;
    const float refractive_color_depth;
    const Texture * roughness_texture;
    const float ior;
};
