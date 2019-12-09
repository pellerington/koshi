#pragma once

#include "Material.h"
#include "MaterialBackLambert.h"
#include "MaterialLambert.h"

class MaterialSubsurface : public Material
{
public:
    MaterialSubsurface(const AttributeVec3f &surface_color_attr, const AttributeFloat &surface_weight);
    std::shared_ptr<Material> instance(const Surface * surface);

    Type get_type() { return Material::Subsurface; }

    bool sample_material(std::vector<MaterialSample> &samples, float sample_reduction = 1.f);
    bool evaluate_material( MaterialSample &sample);

private:
    const AttributeFloat surface_weight_attr;
    float surface_weight;
    std::shared_ptr<MaterialLambert> lambert;
    std::shared_ptr<MaterialBackLambert> back_lambert;
};
