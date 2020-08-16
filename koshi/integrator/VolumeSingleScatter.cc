#include <koshi/integrator/VolumeSingleScatter.h>

#include <koshi/geometry/Volume.h>
#include <koshi/material/MaterialVolume.h>

Vec3f VolumeSingleScatter::integrate(const Intersect * intersect, void * data, Transmittance& transmittance, Resources& resources) const
{
    return VEC3F_ZERO;
}

Vec3f VolumeSingleScatter::shadow(const float& t, const Intersect * intersect, void * data, Resources& resources) const
{
    const Volume * volume = dynamic_cast<const Volume*>(intersect->geometry_data);

    Vec3f shadow = VEC3F_ONES;

    if(!volume || !volume->segment || !volume->material) return shadow;

    if(volume->material->homogenous())
    {
        if(t > volume->segment->t0)
        {
            const float tlen = std::min(volume->segment->t1, t) - volume->segment->t0;
            shadow *= Vec3f::exp(-tlen * volume->segment->max_density);
        }
    }
    else
    {
        // TODO: Test if this causes biasing.
        Random<1> rng = resources.random_service->get_random<1>();
        float inv_tlen = 1.f / intersect->tlen;

        for(Volume::Segment * segment = volume->segment; segment; segment = segment->next)
        {
            float max_density = segment->max_density.max();
            float inv_max_density = 1.f / max_density;

            float tcurr = segment->t0;

            while(true)
            {
                tcurr -= logf(rng.rand()[0]) * inv_max_density;
                if(tcurr > segment->t1 || tcurr > t)
                    break;

                const float d = (tcurr - intersect->t) * inv_tlen;
                Vec3f uvw = volume->uvw_near * d + volume->uvw_far * (1.f - d);
                Vec3f density = volume->material->get_density(uvw, intersect, resources);
                shadow = shadow * (VEC3F_ONES - density * inv_max_density);
            }
        }
    }
    
    return shadow;
}