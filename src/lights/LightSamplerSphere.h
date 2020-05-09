#pragma once

#include <lights/LightSampler.h>
#include <geometry/GeometrySphere.h>

class LightSamplerSphere : public LightSampler
{
public:
    LightSamplerSphere(GeometrySphere * geometry);

    bool sample_light(const uint num_samples, const Intersect * intersect, std::vector<LightSample>& light_samples, Resources& resources);
    float evaluate_light(const Intersect * light_intersect, const Intersect * intersect, Resources& resources);

    bool sample_sa(const uint num_samples, const Intersect * intersect, std::vector<LightSample>& light_samples, Resources& resources);
    float evaluate_sa(const Intersect * light_intersect, const Intersect * intersect, Resources& resources);

    bool sample_area(const uint num_samples, const Intersect * intersect, std::vector<LightSample>& light_samples, Resources& resources);
    float evaluate_area(const Intersect * light_intersect, const Intersect * intersect, Resources& resources);

private:
    GeometrySphere * geometry;
    Light * light;
    
    float area;
};
