#pragma once

#include "Material.h"

class MaterialLambert : public Material
{
public:
    MaterialLambert(const Vec3f &diffuse_color = VEC3F_ZERO);
    std::shared_ptr<Material> instance(const Surface * surface);

    Type get_type() { return Material::Lambert; }

    bool sample_material(std::vector<MaterialSample> &samples, float sample_reduction = 1.f);
    bool evaluate_material(MaterialSample &sample);

private:
    const Vec3f diffuse_color;
};
