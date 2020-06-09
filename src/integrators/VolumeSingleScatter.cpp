#include <integrators/VolumeSingleScatter.h>

#include <geometry/Volume.h>

Vec3f VolumeSingleScatter::integrate(const Intersect * intersect, Transmittance& transmittance, Resources &resources) const
{
    return VEC3F_ZERO;
}

Vec3f VolumeSingleScatter::shadow(const float& t, const Intersect * intersect) const
{
    const Volume * volume = dynamic_cast<const Volume*>(intersect->geometry_data);

    Vec3f shadow = VEC3F_ONES;

    if(!volume) return shadow;

    if(/*integrator_data->homogenous &&*/ volume->segment)
    {
        if(t > volume->segment->t0)
        {
            const float tlen = std::min(volume->segment->t1, t) - volume->segment->t0;
            shadow *= Vec3f::exp(-tlen * volume->segment->max_density);
        }
    }

    return shadow;
}