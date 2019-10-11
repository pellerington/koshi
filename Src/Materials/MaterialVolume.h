#pragma once

#include "Material.h"

class MaterialVolume : public Material
{
public:
    MaterialVolume(const Vec3f &density, const Vec3f &scattering = VEC3F_ZERO, const Vec3f &emission = VEC3F_ZERO);
    std::shared_ptr<Material> instance(const Surface * surface = nullptr);

    Type get_type() { return Material::Volume; }

    bool sample_material(std::vector<MaterialSample> &samples, float sample_reduction = 1.f);
    bool evaluate_material(MaterialSample &sample);

    inline void set_uvw(const Vec3f &_uvw) { uvw = _uvw; };
    virtual const Vec3f get_density();
    virtual const Vec3f get_absorbtion();
    virtual const Vec3f get_scattering();
    const Vec3f get_emission();

    virtual bool is_heterogeneous() { return true; }

private:
    Vec3f uvw;
};
