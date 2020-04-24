#pragma once

#include <integrators/SurfaceIntegrator.h>

class SurfaceIntegratorMaterialSampling : public SurfaceIntegrator
{
    std::vector<SurfaceSample> integrate_surface(
        MaterialInstance * material_instance, Material * material, 
        const Intersect& intersect, const GeometrySurface * surface, 
        Resources& resources) const;

    float evaluate(const Intersect& intersect, const SurfaceSample& sample, Resources& resources);
};
