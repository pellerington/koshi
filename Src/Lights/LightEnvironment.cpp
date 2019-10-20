#include "LightEnvironment.h"

#include <iostream>

LightEnvironment::LightEnvironment(const Vec3f &intensity, std::shared_ptr<Texture> texture)
: Light(Type::Environment, intensity), texture(texture)
{
}

bool LightEnvironment::evaluate_light(const Ray &ray, LightSample &light_sample)
{
    if(!ray.hit && ray.tmax < FLT_MAX)
        return false;

    float theta = acosf(ray.dir.y);
    float phi = atanf((ray.dir.z + EPSILON_F) / (ray.dir.x + EPSILON_F));
    const bool zd = ray.dir.z > 0, xd = ray.dir.x > 0;
    if(!zd) phi += PI;
    if(xd != zd) phi += PI;

    const float u = theta * INV_PI;
    const float v = phi * INV_TWO_PI;

    light_sample.intensity = 1.f;
    if(texture) texture->get_vec3f(v, u, light_sample.intensity);

    light_sample.intensity *= intensity;

    light_sample.position = Vec3f(FLT_MAX);
    light_sample.pdf = 0.f;

    return true;
}

bool LightEnvironment::sample_light(const uint num_samples, const Surface &surface, std::vector<LightSample> &light_samples)
{
    return false;
}
