#pragma once

#include <integrator/SurfaceSampler.h>
#include <light/LightSampler.h>
#include <map>

class SurfaceLightSampler : public SurfaceSampler
{
public:
    void pre_render(Resources& resources);

    struct SurfaceLightSamplerData : public SurfaceSamplerData
    {
        SurfaceLightSamplerData(Resources& resources) : light_data(resources.memory) {}
        Array<const LightSamplerData *> light_data;
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
    std::vector<LightSampler*> lights;
    std::unordered_map<Geometry*, uint> lights_map;
};
