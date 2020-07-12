#pragma once

#include <material/Material.h>

class MaterialLight : public Material
{
public:
    MaterialLight(const Texture * intensity_texture)
    : intensity_texture(intensity_texture)
    {
    }

    virtual Vec3f emission(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources) 
    {
        // TODO: Add custom exposure ect controls.
        return intensity_texture->evaluate<Vec3f>(u, v, w, intersect, resources); 
    }

private:
    const Texture * intensity_texture;
};