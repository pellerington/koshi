#pragma once

#include <koshi/integrator/Integrator.h>
#include <koshi/geometry/SurfaceDistant.h>
#include <koshi/material/Material.h>
#include <koshi/integrator/Transmittance.h>

class LightEvaluator : public Integrator
{
public:
    Vec3f integrate(const Intersect * intersect, void * data, Transmittance& transmittance, Resources& resources) const
    {
        const SurfaceDistant * surface = (const SurfaceDistant*)intersect->geometry_data;
        if(!surface || !surface->material)
            return VEC3F_ZERO;
        return surface->material->emission(surface->u, surface->v, surface->w, intersect, resources) * transmittance.shadow(intersect->t, resources) * surface->opacity;
    }

    virtual Vec3f shadow(const float& t, const Intersect * intersect, void * data, Resources& resources) const
    {
        const SurfaceDistant * surface = (const SurfaceDistant*)intersect->geometry_data;
        return (t > intersect->t) ? (VEC3F_ONES - surface->opacity) : VEC3F_ONES;
    }

private:
    const Vec3f density;
};