#pragma once

#include <koshi/material/Material.h>
#include <koshi/geometry/Geometry.h>

class MaterialLight : public Material
{
public:
    MaterialLight(const Texture * intensity_texture, const bool& normalized)
    : intensity_texture(intensity_texture), normalized(normalized)
    {
    }

    virtual Vec3f emission(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources) 
    {
        // TODO: Add exposure, falloff and other controls here.
        Vec3f color = intensity_texture->evaluate<Vec3f>(u, v, w, intersect, resources);
        if(normalized && intersect)
            color /= intersect->geometry->get_area();
        return color;
    }

    const Texture * get_intensity_texture() const { return intensity_texture; }

private:
    const Texture * intensity_texture;
    const bool normalized;
};