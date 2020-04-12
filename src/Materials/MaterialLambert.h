#pragma once

#include <Materials/Material.h>

class MaterialLambert : public Material
{
public:
    MaterialLambert(const AttributeVec3f &diffuse_color_attr);
    Type get_type() { return Material::Lambert; }

    MaterialInstance * instance(const Surface * surface, Resources &resources);
    struct MaterialInstanceLambert : public MaterialInstance
    {
        Vec3f diffuse_color;
    };

    bool sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, const float sample_reduction, Resources &resources);
    bool evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample);

private:
    const AttributeVec3f diffuse_color_attr;
};
