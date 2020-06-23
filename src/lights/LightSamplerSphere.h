#pragma once

#include <lights/LightSampler.h>
#include <geometry/GeometrySphere.h>

class LightSamplerSphere : public LightSampler
{
public:
    LightSamplerSphere(GeometrySphere * geometry);

    bool sample_light(const uint num_samples, const Surface * surface, std::vector<LightSample>& light_samples, Resources& resources);
    float evaluate_light(const Intersect * intersect, const Surface * surface, Resources& resources);

    bool sample_sa(const uint num_samples, const Surface * surface, std::vector<LightSample>& light_samples, Resources& resources);
    float evaluate_sa(const Intersect * intersect, const Surface * surface, Resources& resources);

    // TODO: Implement solid angle for ellipses.

    bool sample_area(const uint num_samples, const Surface * surface, std::vector<LightSample>& light_samples, Resources& resources);
    float evaluate_area(const Intersect * intersect, const Surface * surface, Resources& resources);

private:
    GeometrySphere * geometry;
    Light * light;
    
    Vec3f center;
    Vec3f radius;
    Vec3f radius_sqr;
    float area;
    bool ellipsoid;
};
