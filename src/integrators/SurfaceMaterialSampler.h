#pragma once

#include <integrators/SurfaceSampler.h>

class SurfaceMaterialSampler : public SurfaceSampler
{
    std::vector<SurfaceSample> integrate_surface(
        const MaterialInstance& material_instance,
        const Intersect * intersect, const Surface * surface, 
        Interiors& interiors, Resources& resources) const;

    float evaluate(const SurfaceSample& sample, 
        const MaterialInstance& material_instance,
        const Intersect * intersect, const Surface * surface, 
        Resources& resources) const;
};
