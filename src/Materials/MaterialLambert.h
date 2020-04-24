#pragma once

#include <Materials/Material.h>

class MaterialLambert : public Material
{
public:
    MaterialLambert(const AttributeVec3f &diffuse_color_attr);

    MaterialInstance * instance(const GeometrySurface * surface, Resources &resources);
    struct MaterialInstanceLambert : public MaterialInstance
    {
        Vec3f diffuse_color;
    };

    bool sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, const float sample_reduction, Resources &resources);
    bool evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample);

private:
    const AttributeVec3f diffuse_color_attr;
};
