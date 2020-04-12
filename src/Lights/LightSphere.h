#pragma once

#include <geometry/ObjectSphere.h>

class LightSphere : public ObjectSphere
{
public:
    LightSphere(const Transform3f &obj_to_world = Transform3f(), std::shared_ptr<Light> light = nullptr, const bool hide_camera = true);

    Type get_type() { return Object::LightSphere; }

    bool sample_light(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources);
    bool evaluate_light(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources);

    bool sample_sa(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources);
    bool evaluate_sa(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources);

    bool sample_area(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources);
    bool evaluate_area(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources);

private:
    float area;
};
