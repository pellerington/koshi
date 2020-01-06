#pragma once

#include "Material.h"

class MaterialLambert : public Material
{
public:
    MaterialLambert(const AttributeVec3f &diffuse_color_attr);
    Type get_type() { return Material::Lambert; }

    std::shared_ptr<MaterialInstance> instance(const Surface * surface);
    struct MaterialInstanceLambert : public MaterialInstance
    {
        Vec3f diffuse_color;
    };

    bool sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, RNG &rng, const float sample_reduction);
    bool evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample);

private:
    const AttributeVec3f diffuse_color_attr;
};
