#pragma once

#include <material/Material.h>
#include <material/Fresnel.h>

class MaterialDielectric : public Material
{
public:
    MaterialDielectric(const Texture * reflective_color_texture,
                       const Texture * refractive_color_texture,
                       const float& refractive_color_depth, 
                       const Texture * roughness_texture,
                       const float& ior);

    MaterialInstance instance(const Surface * surface, const Intersect * intersect, Resources &resources);

private:
    const Texture * reflective_color_texture;
    const Texture * refractive_color_texture;
    const float refractive_color_depth;
    const Texture * roughness_texture;
    const float ior;
};
