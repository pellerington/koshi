#include <koshi/integrator/Integrator.h>
#include <koshi/integrator/Transmittance.h>

Vec3f Integrator::shade(const IntersectList * intersects, Resources& resources)
{
    if(intersects->empty())
        return VEC3F_ZERO;

    Vec3f color = VEC3F_ZERO;

    // TODO: STORE data straight into the transmittance.
    Array<void*> data(resources.memory);
    for(uint i = 0; i < intersects->size(); i++)
        if(intersects->get(i)->integrator)
            data.push(intersects->get(i)->integrator->pre_integrate(intersects->get(i), resources));

    Transmittance transmittance(intersects, data);

    for(uint i = 0; i < intersects->size(); i++)
        if(intersects->get(i)->integrator)
            color += intersects->get(i)->integrator->integrate(intersects->get(i), data[i], transmittance, resources);

    return color;
}

Transmittance Integrator::shadow(const IntersectList * intersects, Resources& resources)
{
    Array<void*> data(resources.memory);
    for(uint i = 0; i < intersects->size(); i++)
        if(intersects->get(i)->integrator)
            data.push(intersects->get(i)->integrator->pre_integrate(intersects->get(i), resources));

    return Transmittance(intersects, data);
}
