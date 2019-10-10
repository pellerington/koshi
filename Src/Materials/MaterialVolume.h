#pragma once

#include "Material.h"

class MaterialVolume : public Material
{
public:
    MaterialVolume();
    std::shared_ptr<Material> instance(const Surface &surface);

    Type get_type() { return Material::Volume; }

    bool sample_material(const Surface &surface, std::vector<MaterialSample> &samples, float sample_reduction = 1.f);
    bool evaluate_material(const Surface &surface, MaterialSample &sample);

    inline void set_uvw(const Vec3f &_uvw) { uvw = _uvw; };
    virtual const Vec3f get_density();
    virtual const Vec3f get_absorbtion();
    virtual const Vec3f get_scattering();
    const Vec3f get_emission();

    virtual bool is_heterogeneous();

private:
    Vec3f uvw;
};
