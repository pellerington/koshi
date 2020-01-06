#pragma once

#include "Material.h"
#include "MaterialBackLambert.h"
#include "MaterialLambert.h"

class MaterialSubsurface : public Material
{
public:
    MaterialSubsurface(const AttributeVec3f &surface_color_attr, const AttributeFloat &surface_weight);
    Type get_type() { return Material::Subsurface; }

    std::shared_ptr<MaterialInstance> instance(const Surface * surface);
    struct MaterialInstanceSubsurface : public MaterialInstance
    {
        float surface_weight;
        MaterialLambert::MaterialInstanceLambert lambert_instance;
        MaterialBackLambert::MaterialInstanceBackLambert back_lambert_instance;
    };

    bool sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, RNG &rng, const float sample_reduction);
    bool evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample);

private:
    const AttributeVec3f surface_color_attr;
    const AttributeFloat surface_weight_attr;

    MaterialLambert lambert;
    MaterialBackLambert back_lambert;
};
