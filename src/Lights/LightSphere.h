#pragma once

#include <geometry/GeometrySphere.h>

class LightSphere : public GeometrySphere
{
public:
    LightSphere(const Transform3f &obj_to_world = Transform3f(), std::shared_ptr<Light> light = nullptr, const bool hide_camera = true);

    bool sample_light(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources);
    bool evaluate_light(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources);

    bool sample_sa(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources);
    bool evaluate_sa(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources);

    bool sample_area(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources);
    bool evaluate_area(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources);
};
