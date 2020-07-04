#pragma once

#include <light/LightSampler.h>
#include <geometry/GeometryArea.h>

class LightSamplerArea : public LightSampler
{
public:
    LightSamplerArea(GeometryArea * geometry);

    struct LightSamplerDataArea : public LightSamplerData
    {
        Random<2> rng;
    };
    const LightSamplerData * pre_integrate(const Surface * surface, Resources& resources) const;

    bool sample(LightSample& sample, const LightSamplerData * data, Resources& resources) const;
    float evaluate(const Intersect * intersect, const LightSamplerData * data, Resources& resources) const;

    LightType get_light_type() const { return LightType::AREA; }

private:
    GeometryArea * geometry;
    Light * light;
    bool double_sided;
    Vec3f normal;
    float area;
};
