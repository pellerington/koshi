#pragma once

#include <Math/Types.h>
#include <material/Material.h>
#include <geometry/Volume.h>

class MaterialVolume  : public Object
{
public:
    MaterialVolume
    (
        const AttributeVec3f& density,  const std::string& density_name,
        const AttributeVec3f& scatter,  const std::string& scatter_name,
        const AttributeVec3f& emission, const std::string& emission_name,
        const AttributeFloat& anistropy
    )
    : density(density), scatter(scatter), emission(emission), 
      density_name(density_name), scatter_name(scatter_name), emission_name(emission_name),
      anistropy(anistropy)
    {}

    bool homogenous() const { return density_name == "" && density.constant(); }
    bool has_scatter() const { return scatter_name == "" && scatter.null(); }
    bool has_emission() const { return emission_name == "" && emission.null(); }

    virtual Vec3f get_density(const Vec3f& uvw, const Volume * volume, const Intersect * intersect, Resources& resources) const;
    virtual Vec3f get_scatter(const Vec3f& uvw, const Volume * volume, const Intersect * intersect, Resources& resources) const;
    virtual Vec3f get_emission(const Vec3f& uvw, const Volume * volume, const Intersect * intersect, Resources& resources) const;

    virtual MaterialLobe * get_lobe(const Vec3f& uvw, const Volume * volume, const Intersect * intersect, Resources& resources) const;

private:
    const AttributeVec3f density, scatter, emission;
    const std::string density_name, scatter_name, emission_name;
    const AttributeFloat anistropy;
};