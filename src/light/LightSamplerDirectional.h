#pragma once

#include <light/LightSampler.h>
#include <geometry/GeometryDirectional.h>

// TODO: Add a GeometryDirectional class and make it so we can increase the cone of influence
class LightSamplerDirectional : public LightSampler
{
public:
    LightSamplerDirectional(GeometryDirectional * geometry);

    void pre_render(Resources& resources);

    struct LightSamplerDataDirectional : public LightSamplerData
    {
        Random<2> rng;
    };
    const LightSamplerData * pre_integrate(const Surface * surface, Resources& resources) const;

    bool sample(LightSample& sample, const LightSamplerData * data, Resources& resources) const;
    float evaluate(const Intersect * intersect, const LightSamplerData * data, Resources& resources) const;

    LightType get_light_type() const { return  geometry->get_phi_max() > EPSILON_F ? LightType::AREA : LightType::POINT; }

private:
    GeometryDirectional * geometry;
    Material * material;
};
