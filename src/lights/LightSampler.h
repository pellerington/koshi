#pragma once

#include <base/Object.h>
#include <Util/Resources.h>
#include <Math/Types.h>

#include <vector>

struct LightSample {
    Vec3f position;
    Vec3f intensity;
    float pdf = 0.f;
};

class LightSampler : public Object
{
public:
    virtual bool sample_light(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample>& light_samples, Resources& resources) = 0;
    virtual bool evaluate_light(const Surface& intersect, const Vec3f * pos, const Vec3f * pfar, LightSample& light_sample, Resources& resources) = 0;
};