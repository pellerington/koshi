#pragma once

#include "Material.h"

class MaterialLambert : public Material
{
public:
    MaterialLambert(const AttributeVec3f &diffuse_color_attr);
    std::shared_ptr<Material> instance(const Surface * surface, RNG &rng);

    Type get_type() { return Material::Lambert; }

    bool sample_material(std::vector<MaterialSample> &samples, float sample_reduction = 1.f);
    bool evaluate_material(MaterialSample &sample);

private:
    const AttributeVec3f diffuse_color_attr;
    Vec3f diffuse_color;
};
