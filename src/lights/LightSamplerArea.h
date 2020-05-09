#pragma once

#include <lights/LightSampler.h>
#include <geometry/GeometryArea.h>

class LightSamplerArea : public LightSampler
{
public:
    LightSamplerArea(GeometryArea * geometry);

    bool sample_light(const uint num_samples, const Intersect * intersect, std::vector<LightSample>& light_samples, Resources& resources);
    float evaluate_light(const Intersect * light_intersect, const Intersect * intersect, Resources &resources);
private:
    GeometryArea * geometry;
    Light * light;

    bool double_sided;
    float area;
};
