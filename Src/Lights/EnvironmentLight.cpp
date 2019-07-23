#include "EnvironmentLight.h"

#include <iostream>

EnvironmentLight::EnvironmentLight(Vec3f intensity, std::shared_ptr<Texture> texture)
: Light(Type::Environment, intensity), texture(texture)
{
}

bool EnvironmentLight::evaluate_light(const Ray &ray, Vec3f &light, float* pdf)
{
    light = 1.f;

    float theta = acosf(ray.dir.y());
    float phi = atanf((ray.dir.z() + EPSILON) / (ray.dir.x() + EPSILON));
    bool zd = ray.dir.z() > 0, xd = ray.dir.x() > 0;
    if(!zd) phi += PI;
    if(xd != zd) phi += PI;

    float u = theta * INV_PI;
    float v = phi * INV_TWO_PI;

    if(texture)
        texture->get_vec3f(v, u, light);

    light = light * intensity;

    return true;
}

bool EnvironmentLight::sample_light(const uint num_samples, const Surface &surface, std::deque<SrfSample> &srf_samples)
{
    return false;
}
