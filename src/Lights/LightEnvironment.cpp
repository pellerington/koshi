#include "LightEnvironment.h"

#include <iostream>

LightEnvironment::LightEnvironment(std::shared_ptr<Light> light, std::shared_ptr<Texture> texture)
: Object(Transform3f(), light ? light : std::shared_ptr<Light>(new Light(VEC3F_ZERO))), texture(texture)
{
    set_null_rtc_geometry();
}

bool LightEnvironment::sample_light(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources)
{
    return false;
}

bool LightEnvironment::evaluate_light(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources)
{
    if(intersect.hit)
        return false;

    float theta = acosf(intersect.wi.y);
    float phi = atanf((intersect.wi.z + EPSILON_F) / (intersect.wi.x + EPSILON_F));
    const bool zd = intersect.wi.z > 0, xd = intersect.wi.x > 0;
    if(!zd) phi += PI;
    if(xd != zd) phi += PI;

    const float u = theta * INV_PI;
    const float v = phi * INV_TWO_PI;

    light_sample.intensity = VEC3F_ONES;
    if(texture)
        light_sample.intensity = texture->get_vec3f(v, u, 0.f, resources);

    light_sample.intensity *= light->get_emission();

    light_sample.position = Vec3f(FLT_MAX);
    light_sample.pdf = 0.f;

    return true;
}
