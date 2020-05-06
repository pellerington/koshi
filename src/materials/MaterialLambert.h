#pragma once

#include <materials/Material.h>

template<bool FRONT>
struct MaterialLobeLambert : public MaterialLobe
{
    bool sample(MaterialSample& sample, Resources& resources) const;
    bool evaluate(MaterialSample& sample, Resources& resources) const;

    Type type() const { return Type::Diffuse; }
};

template<bool FRONT>
class MaterialLambert : public Material
{
public:
    MaterialLambert(const AttributeVec3f &diffuse_color_attr);
    MaterialInstance instance(const GeometrySurface * surface, Resources &resources);
private:
    const AttributeVec3f diffuse_color_attr;
};

template class MaterialLambert<true>;
template class MaterialLambert<false>;