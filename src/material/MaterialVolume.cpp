#include <material/MaterialVolume.h>

Vec3f MaterialVolume::get_density(const Vec3f& uvw, const Intersect * intersect, Resources& resources) const
{
    return density.get_value(uvw.u, uvw.v, uvw.w, resources);
}

Vec3f MaterialVolume::get_scatter(const Vec3f& uvw, const Intersect * intersect, Resources& resources) const
{
    return VEC3F_ZERO;
}

Vec3f MaterialVolume::get_emission(const Vec3f& uvw, const Intersect * intersect, Resources& resources) const
{
    return VEC3F_ZERO;
}

MaterialLobe * MaterialVolume::get_lobe(const Vec3f& uvw, const Intersect * intersect, Resources& resources) const
{
    return nullptr;
}
