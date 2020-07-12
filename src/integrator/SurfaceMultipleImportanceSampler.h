#pragma once

#include <integrator/SurfaceSampler.h>

class SurfaceMultipleImportanceSampler : public SurfaceSampler
{
public:
    void pre_render(Resources& resources);

    struct SurfaceMultipleImportanceSamplerData : public SurfaceSamplerData
    {
        SurfaceMultipleImportanceSamplerData(Resources& resources) : integrator_data(resources.memory) {}
        Array<SurfaceSamplerData *> integrator_data;
    };
    IntegratorData * pre_integrate(const Intersect * intersect, Resources& resources);

    void scatter_surface(
        Array<SurfaceSample>& samples,
        const MaterialLobes& lobes, 
        const Intersect * intersect, SurfaceSamplerData * data,
        Interiors& interiors, Resources& resources) const;

    float evaluate(const SurfaceSample& sample, 
        const MaterialLobes& lobes,
        const Intersect * intersect, SurfaceSamplerData * data,
        Resources& resources) const;

private:
    std::vector<SurfaceSampler*> integrators;
};
