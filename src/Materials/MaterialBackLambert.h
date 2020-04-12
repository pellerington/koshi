#pragma once

#include <Materials/Material.h>

class MaterialBackLambert : public Material
{
public:
    MaterialBackLambert(const AttributeVec3f &diffuse_color_attr);
    Type get_type() { return Material::BackLambert; }

    MaterialInstance * instance(const Surface * surface, Resources &resources);
    struct MaterialInstanceBackLambert : public MaterialInstance
    {
        Vec3f diffuse_color;
    };

    bool sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, const float sample_reduction, Resources &resources);
    bool evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample);

private:

    const AttributeVec3f diffuse_color_attr;
};
