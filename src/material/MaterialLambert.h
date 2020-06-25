#pragma once

#include <material/Material.h>

template<bool REFLECT>
struct MaterialLobeLambert : public MaterialLobe
{
    bool sample(MaterialSample& sample, Resources& resources) const;
    Vec3f weight(const Vec3f& wo, Resources& resources) const;
    float pdf(const Vec3f& wo, Resources& resources) const;
    ScatterType get_scatter_type() const { return ScatterType::DIFFUSE; }
    Hemisphere get_hemisphere() const { return REFLECT ? Hemisphere::FRONT : Hemisphere::BACK; }
};

template<bool REFLECT>
class MaterialLambert : public Material
{
public:
    MaterialLambert(const AttributeVec3f &color_attr);
    MaterialInstance instance(const Surface * surface, const Intersect * intersect, Resources &resources);
private:
    const AttributeVec3f color_attr;
};

template struct MaterialLobeLambert<true>;
template struct MaterialLobeLambert<false>;

template class MaterialLambert<true>;
template class MaterialLambert<false>;