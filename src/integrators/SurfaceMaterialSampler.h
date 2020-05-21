#pragma once

#include <integrators/IntegratorSurface.h>

class SurfaceMaterialSampler : public IntegratorSurface
{
    std::vector<SurfaceSample> integrate_surface(
        const MaterialInstance& material_instance,
        const Intersect * intersect, const GeometrySurface * surface, 
        InteriorMedium& interiors, Resources& resources) const;

    float evaluate(const SurfaceSample& sample, 
        const MaterialInstance& material_instance,
        const Intersect * intersect, const GeometrySurface * surface, 
        Resources& resources) const;
};
