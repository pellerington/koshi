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
    virtual bool sample_light(const uint num_samples, const Intersect& intersect, std::vector<LightSample>& light_samples, Resources& resources) = 0;
    virtual bool evaluate_light(const Intersect& light_intersect, const Intersect& intersect, LightSample& light_sample, Resources& resources) = 0;
};