#include "RectangleLight.h"

#include <iostream>
#include "../Math/RNG.h"

RectangleLight::RectangleLight(Vec3f position, Vec3f u, Vec3f v, Vec3f intensity, bool double_sided)
: Light(Type::Rectangle, intensity), position(position), u(u), v(v), double_sided(double_sided)
{
    normal = u.normalized().cross(v.normalized());
    area = u.norm() * v.norm();
}

bool RectangleLight::evaluate_light(const Ray &ray, Vec3f &light, float* pdf)
{
    float t = (position - ray.o).dot(normal) / (ray.dir.dot(normal));

    Vec3f light_point = ray.o + t * ray.dir;
    Vec3f dir = light_point - ray.o;

    if(ray.dir.dot(-normal) < 0 && !double_sided)
        return false;

    if(t > ray.t || t < 0)
        return false;

    float tu = u.normalized().dot(light_point - position);
    float tv = v.normalized().dot(light_point - position);
    if(tu < 0.f || tu > u.norm() || tv < 0.f || tv > v.norm())
        return false;

    if(pdf)
        *pdf = dir.squaredNorm() / (area * normal.dot(-dir));

    light = intensity;
    return true;
}

bool RectangleLight::sample_light(const uint num_samples, const Surface &surface, std::deque<SrfSample> &srf_samples) //Should sample a point so that bidirectional works
{
    std::vector<Vec2f> rnd;
    RNG::StratifiedRand(num_samples, rnd);

    for(uint i = 0; i < num_samples; i++)
    {
        Vec3f dir = ((position + rnd[i][0]*u + rnd[i][1]*v) - surface.position);

        if(dir.dot(surface.normal) < 0.f)
            continue;
        if(dir.dot(-normal) < 0.f && !double_sided)
            continue;

        srf_samples.emplace_back();
        SrfSample &srf_sample = srf_samples.back();
        srf_sample.type = SrfSample::Light;

        srf_sample.wo = dir.normalized();
        srf_sample.pdf = dir.squaredNorm() / (area * fabs(normal.dot(-dir)));
    }

    return true;
}
