#pragma once

#include "Material.h"

class MaterialBackLambert : public Material
{
public:
    MaterialBackLambert(const AttributeVec3f &diffuse_color_attr);
    Type get_type() { return Material::BackLambert; }

    std::shared_ptr<MaterialInstance> instance(const Surface * surface);
    struct MaterialInstanceBackLambert : public MaterialInstance
    {
        Vec3f diffuse_color;
    };

    bool sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, RNG &rng, const float sample_reduction);
    bool evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample);

private:

    const AttributeVec3f diffuse_color_attr;
};
