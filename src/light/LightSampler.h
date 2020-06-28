#pragma once

#include <base/Object.h>
#include <Util/Resources.h>
#include <Math/Types.h>
#include <vector>
#include <intersection/Intersect.h>

struct LightSample {
    Vec3f position;
    Vec3f intensity;
    float pdf = 0.f;
};

struct LightSamplerData
{
    const Surface * surface;
    ~LightSamplerData();
};

class LightSampler : public Object
{
public:
    // TODO: We need to handle equiangular volume sampling, and general (no surface) point sample.
    virtual const LightSamplerData * pre_integrate(const Surface * surface, Resources& resources) const = 0;
    virtual bool sample(LightSample& sample, const LightSamplerData * data, Resources& resources) const = 0;
    virtual float evaluate(const Intersect * intersect, const LightSamplerData * data, Resources& resources) const = 0;

    enum LightType { AREA, POINT };
    virtual LightType get_light_type() const = 0;
};