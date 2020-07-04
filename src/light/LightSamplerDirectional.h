#pragma once

#include <light/LightSampler.h>
#include <geometry/Geometry.h>

// TODO: Add a GeometryDirectional class and make it so we can increase the cone of influence
class LightSamplerDirectional : public LightSampler
{
public:
    LightSamplerDirectional(Geometry * geometry);

    struct LightSamplerDataDirectional : public LightSamplerData
    {
        Random<2> rng;
    };
    const LightSamplerData * pre_integrate(const Surface * surface, Resources& resources) const;

    bool sample(LightSample& sample, const LightSamplerData * data, Resources& resources) const;
    float evaluate(const Intersect * intersect, const LightSamplerData * data, Resources& resources) const
    {
        // TODO: Fill this in once we can intersect GeometryDirectional
        return 0.f;
    }

    LightType get_light_type() const { return /* angle > EPSILON ? LightType::AREA : */ LightType::POINT; }

private:
    Geometry * geometry;
    Light * light;

    Vec3f direction;
};
