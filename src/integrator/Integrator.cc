#include <integrator/Integrator.h>

Vec3f Integrator::shade(const IntersectList * intersects, Resources &resources)
{
    if(intersects->empty())
        return VEC3F_ZERO;

    Vec3f color = VEC3F_ZERO;

    // TODO: also add pre_render here;

    Transmittance transmittance(intersects);
    for(uint i = 0; i < intersects->size(); i++)
        if(intersects->get(i)->integrator)
            color += intersects->get(i)->integrator->integrate(intersects->get(i), transmittance, resources);

    return color;
}

Transmittance Integrator::shadow(const IntersectList * intersects, Resources &resources)
{
    // TODO: also add pre_render here;

    return Transmittance(intersects);
}
