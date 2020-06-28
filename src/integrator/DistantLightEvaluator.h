#pragma once

#include <integrator/Integrator.h>
#include <geometry/SurfaceDistant.h>

class DistantLightEvaluator : public Integrator
{
public:
    Vec3f integrate(const Intersect * intersect, IntegratorData * data, Transmittance& transmittance, Resources& resources) const
    {
        const SurfaceDistant * distant = (const SurfaceDistant*)intersect->geometry_data;
        Light * light = intersect->geometry->get_attribute<Light>("light");
        return light->get_intensity(distant->u, distant->v, 0.f, intersect, resources) *  transmittance.shadow(intersect->t, resources) * distant->opacity;
    }

    virtual Vec3f shadow(const float& t, const Intersect * intersect, IntegratorData * data, Resources& resources) const
    {
        return VEC3F_ONES;
    }

private:
    const Vec3f density;
};