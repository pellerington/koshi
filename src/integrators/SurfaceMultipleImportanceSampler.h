#pragma once

#include <integrators/SurfaceSampler.h>

class SurfaceMultipleImportanceSampler : public SurfaceSampler
{
public:
    void pre_render(Scene * scene);

    std::vector<SurfaceSample> integrate_surface(
        const MaterialInstance& material_instance, 
        const Intersect * intersect, const GeometrySurface * surface,
        Interiors& interiors, Resources& resources) const;

    float evaluate(const SurfaceSample& sample, 
        const MaterialInstance& material_instance,
        const Intersect * intersect, const GeometrySurface * surface, 
        Resources& resources) const;

private:
    std::vector<SurfaceSampler*> integrators;
};
