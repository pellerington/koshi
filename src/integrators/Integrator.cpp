#include <integrators/Integrator.h>

Vec3f Integrator::shade(const IntersectList * intersects, Resources &resources)
{
    if(intersects->empty())
        return VEC3F_ZERO;

    Vec3f color = VEC3F_ZERO;

    IntegratorList * integrators = nullptr;
    for(const Intersect * intersect = intersects->get(0); intersect; intersect = intersect->next)
    {
        if(intersect->integrator)
        {
            // TODO: also add pre_render here;
            integrators = resources.memory.create<IntegratorList>(integrators);
            integrators->intersect = intersect;
            integrators->integrator = intersect->integrator;
        }
    }

    Transmittance transmittance(intersects, integrators);
    for(const IntegratorList * integrator = integrators; integrator; integrator = integrator->next)
        color += integrator->intersect->opacity * integrator->integrator->integrate(integrator->intersect, transmittance, resources);

    return color;
}

Transmittance Integrator::shadow(const IntersectList * intersects, Resources &resources)
{
    IntegratorList * integrators = nullptr;
    for(const Intersect * intersect = intersects->get(0); intersect; intersect = intersect->next)
    {
        if(intersect->integrator)
        {
            // TODO: also add pre_render here;
            integrators = resources.memory.create<IntegratorList>(integrators);
            integrators->intersect = intersect;
            integrators->integrator = intersect->integrator;
        }
    }

    return Transmittance(intersects, integrators);   
}
