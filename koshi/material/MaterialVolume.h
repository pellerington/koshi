#pragma once

#include <koshi/math/Types.h>
#include <koshi/texture/Texture.h>
#include <koshi/material/Material.h>
#include <koshi/geometry/Volume.h>
#include <koshi/material/MaterialHenyeyGreenstein.h>

class MaterialVolume  : public Object
{
public:
    MaterialVolume
    (
        const Texture * density_texture,  const std::string& density_name,
        const Texture * scatter_texture,  const std::string& scatter_name,
        const Texture * emission_texture, const std::string& emission_name,
        const Texture * anistropy_texture
    )
    : density_texture(density_texture), scatter_texture(scatter_texture), emission_texture(emission_texture),
      density_name(density_name), scatter_name(scatter_name), emission_name(emission_name)
    {
        surface_material = new MaterialHenyeyGreenstein(anistropy_texture);
    }

    bool homogenous() const { return density_name == "" && density_texture->delta() == VEC3F_ONES; }
    bool has_scatter() const { return scatter_name != "" || !scatter_texture->null(); }
    bool has_emission() const { return emission_name != "" || !emission_texture->null(); }

    const Texture * get_density_texture() { return density_texture;}

    virtual Vec3f get_density(const Vec3f& uvw, const Intersect * intersect, Resources& resources) const;
    virtual Vec3f get_scatter(const Vec3f& uvw, const Intersect * intersect, Resources& resources) const;
    virtual Vec3f get_emission(const Vec3f& uvw, const Intersect * intersect, Resources& resources) const;

    // TODO: Have MaterialVolume extend Material so we don't need this.
    virtual Material * get_surface_material() const { return surface_material; }

private:
    const Texture * density_texture;
    const Texture * scatter_texture;
    const Texture * emission_texture;
    const std::string density_name, scatter_name, emission_name;

    // TODO: Delete this in ~MaterialVolume()
    Material * surface_material;
};