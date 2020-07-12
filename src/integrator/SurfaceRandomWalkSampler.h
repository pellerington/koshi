#pragma once

#include <integrator/SurfaceSampler.h>
#include <intersection/IntersectCallbacks.h>

class SurfaceRandomWalkSampler : public SurfaceSampler
{
public:
    void scatter_surface(
        Array<SurfaceSample>& samples,
        const MaterialLobes& lobes,
        const Intersect * intersect, SurfaceSamplerData * data, 
        Interiors& interiors, Resources& resources) const;

    float evaluate(const SurfaceSample& sample, 
        const MaterialLobes& lobes,
        const Intersect * intersect, SurfaceSamplerData * data, 
        Resources& resources) const
    {
        return 0.f;
    }

private:
    static void post_intersection_callback(IntersectList * intersects, void * data, Resources& resources);
    static IntersectionCallbacks intersection_callback;
};
