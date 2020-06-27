#pragma once

#include <integrator/SurfaceSampler.h>

class SurfaceMultipleImportanceSampler : public SurfaceSampler
{
public:
    void pre_render(Resources& resources);

    std::vector<SurfaceSample> integrate_surface(
        const MaterialInstance& material_instance, 
        const Intersect * intersect, const Surface * surface,
        Interiors& interiors, Resources& resources) const;

    float evaluate(const SurfaceSample& sample, 
        const MaterialInstance& material_instance,
        const Intersect * intersect, const Surface * surface, 
        Resources& resources) const;

private:
    std::vector<SurfaceSampler*> integrators;
};
