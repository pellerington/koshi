#include "LightRectangle.h"

#include <iostream>
#include "../Math/RNG.h"

LightRectangle::LightRectangle(const Vec3f &position, const Vec3f &u, const Vec3f &v, const Vec3f &intensity, const bool double_sided)
: Light(Type::Rectangle, intensity), position(position), u(u), v(v), normal(u.normalized().cross(v.normalized()))
, area(u.length() * v.length()), double_sided(double_sided)
{
}

bool LightRectangle::evaluate_light(const Ray &ray, LightSample &light_sample)
{
    const float t = (position - ray.pos).dot(normal) / (ray.dir.dot(normal));

    light_sample.position = ray.get_position(t);
    const Vec3f dir = light_sample.position - ray.pos;

    if(ray.dir.dot(-normal) < 0 && !double_sided)
        return false;

    if(t > ray.tmax || t < 0)
        return false;

    const float tu = u.normalized().dot(light_sample.position - position);
    const float tv = v.normalized().dot(light_sample.position - position);
    if(tu < 0.f || tu > u.length() || tv < 0.f || tv > v.length())
        return false;

    light_sample.pdf = dir.sqr_length() / (area * normal.dot(-dir));
    light_sample.intensity = intensity;

    return true;
}

bool LightRectangle::sample_light(const uint num_samples, const Surface &surface, std::vector<LightSample> &light_samples)
{
    std::vector<Vec2f> rnd;
    RNG::Rand2d(num_samples, rnd);

    for(uint i = 0; i < rnd.size(); i++)
    {
        const Vec3f point = (position + rnd[i][0]*u + rnd[i][1]*v);
        const Vec3f dir = (point - surface.position);

        const float cos_theta = dir.dot(-normal);
        if(cos_theta < 0.f && !double_sided)
            continue;

        light_samples.emplace_back();
        LightSample &light_sample = light_samples.back();

        light_sample.position = point;
        light_sample.intensity = intensity;
        light_sample.pdf = dir.sqr_length() / (area * (fabs(cos_theta) + EPSILON_F)); // Do the solid angle pdf in a different function? just do actual pdf here (ie textured)
    }

    return true;
}
