#include <material/MaterialVolume.h>

Vec3f MaterialVolume::get_density(const Vec3f& uvw, const Intersect * intersect, Resources& resources) const
{
    return density_texture->evaluate<Vec3f>(uvw.u, uvw.v, uvw.w, intersect, resources);
}

Vec3f MaterialVolume::get_scatter(const Vec3f& uvw, const Intersect * intersect, Resources& resources) const
{
    return VEC3F_ZERO;
}

Vec3f MaterialVolume::get_emission(const Vec3f& uvw, const Intersect * intersect, Resources& resources) const
{
    return VEC3F_ZERO;
}

MaterialLobes MaterialVolume::instance(const Vec3f& uvw, const Intersect * intersect, Resources& resources) const
{
    return MaterialLobes(resources.memory);
}
