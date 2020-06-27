#pragma once

#include <integrator/Integrator.h>
#include <geometry/SurfaceDistant.h>

class DistantLightEvaluator : public Integrator
{
public:
    Vec3f integrate(const Intersect * intersect, Transmittance& transmittance, Resources &resources) const
    {
        const SurfaceDistant * distant = (const SurfaceDistant*)intersect->geometry_data;

        // Add light contribution.
        Light * light = intersect->geometry->get_attribute<Light>("light");
        return light->get_intensity(distant->u, distant->v, 0.f, intersect, resources) *  transmittance.shadow(intersect->t, resources) * distant->opacity;
    }

    virtual Vec3f shadow(const float& t, const Intersect * intersect, Resources &resources) const
    {
        // TODO: Treat surface and distant seperatly?
        return VEC3F_ONES;
    }

private:
    const Vec3f density;
};