#pragma once

#include <koshi/math/Types.h>
#include <koshi/texture/Texture.h>
#include <koshi/material/Material.h>
#include <koshi/geometry/Volume.h>

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
    : density_texture(density_texture), scatter_texture(scatter_texture), 
      emission_texture(emission_texture), anistropy_texture(anistropy_texture),
      density_name(density_name), scatter_name(scatter_name), emission_name(emission_name)
    {}

    bool homogenous() const { return density_name == "" && density_texture->delta() == VEC3F_ONES; }
    bool has_scatter() const { return scatter_name == "" && scatter_texture->null(); }
    bool has_emission() const { return emission_name == "" && emission_texture->null(); }

    const Texture * get_density_texture() { return density_texture;}

    virtual Vec3f get_density(const Vec3f& uvw, const Intersect * intersect, Resources& resources) const;
    virtual Vec3f get_scatter(const Vec3f& uvw, const Intersect * intersect, Resources& resources) const;
    virtual Vec3f get_emission(const Vec3f& uvw, const Intersect * intersect, Resources& resources) const;

    virtual MaterialLobes instance(const Vec3f& uvw, const Intersect * intersect, Resources& resources) const;

private:
    const Texture * density_texture;
    const Texture * scatter_texture;
    const Texture * emission_texture;
    const Texture * anistropy_texture;
    const std::string density_name, scatter_name, emission_name;
};