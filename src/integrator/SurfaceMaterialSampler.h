#pragma once

#include <integrator/SurfaceSampler.h>

class SurfaceMaterialSampler : public SurfaceSampler
{
    void scatter_surface(
        Array<SurfaceSample>& samples,
        const MaterialInstance& material_instance,
        const Intersect * intersect, SurfaceSamplerData * data,
        Interiors& interiors, Resources& resources) const;

    float evaluate(const SurfaceSample& sample, 
        const MaterialInstance& material_instance,
        const Intersect * intersect, SurfaceSamplerData * data,
        Resources& resources) const;
};
