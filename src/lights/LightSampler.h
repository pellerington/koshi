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

class LightSampler : public Object
{
public:
    // virtual bool sample_light(LightSample& sample, LightData??? * data, Resources& resources) { return false; }
    // virtual bool sample_light(LightSample& sample, const GeometrySurface * surface, LightData??? * data, Resources& resources) { return false; }
    
    virtual bool sample_light(const uint num_samples, const GeometrySurface * surface, std::vector<LightSample>& light_samples, Resources& resources) = 0;
    virtual float evaluate_light(const Intersect * intersect, const GeometrySurface * surface, Resources& resources) = 0;
};