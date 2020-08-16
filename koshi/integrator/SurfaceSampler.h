#pragma once

#include <koshi/integrator/Integrator.h>
#include <koshi/material/Material.h>
#include <koshi/light/LightSampler.h>
#include <koshi/math/Color.h>

#define SAMPLES_PER_HEMISPHERE 128

class SurfaceSampler : public Integrator
{
public:
    void pre_render(Resources& resources);

    struct SurfaceSamplerData { const Surface * surface; };
    
    void * pre_integrate(const Intersect * intersect, Resources& resources);

    Vec3f integrate(const Intersect * intersect, void * data, Transmittance& transmittance, Resources& resources) const;

    Vec3f shadow(const float& t, const Intersect * intersect, void * data, Resources& resources) const;

private:
    float min_quality;

    struct LightSamplerItem { LightSampler * sampler; uint id; };
    robin_hood::unordered_map<Geometry*, LightSamplerItem> lights;
};