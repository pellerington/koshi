#include "RectangleLight.h"

#include <iostream>
#include "../Math/RNG.h"

RectangleLight::RectangleLight(const Vec3f &position, const Vec3f &u, const Vec3f &v, const Vec3f &intensity, const bool double_sided)
: Light(Type::Rectangle, intensity), position(position), u(u), v(v), normal(u.normalized().cross(v.normalized()))
, area(u.length() * v.length()), double_sided(double_sided)
{
}

bool RectangleLight::evaluate_light(const Ray &ray, Vec3f &light, float * pdf)
{
    const float t = (position - ray.o).dot(normal) / (ray.dir.dot(normal));

    const Vec3f light_point = ray.o + t * ray.dir;
    const Vec3f dir = light_point - ray.o;

    if(ray.dir.dot(-normal) < 0 && !double_sided)
        return false;

    if(t > ray.t || t < 0)
        return false;

    const float tu = u.normalized().dot(light_point - position);
    const float tv = v.normalized().dot(light_point - position);
    if(tu < 0.f || tu > u.length() || tv < 0.f || tv > v.length())
        return false;

    if(pdf)
        *pdf = dir.sqr_length() / (area * normal.dot(-dir));

    light = intensity;
    return true;
}

bool RectangleLight::sample_light(const uint num_samples, const Surface &surface, std::deque<LightSample> &light_samples)
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
