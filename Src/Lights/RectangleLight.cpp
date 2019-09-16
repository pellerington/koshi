#include "RectangleLight.h"

#include <iostream>
#include "../Math/RNG.h"

RectangleLight::RectangleLight(Vec3f position, Vec3f u, Vec3f v, Vec3f intensity, bool double_sided)
: Light(Type::Rectangle, intensity), position(position), u(u), v(v), double_sided(double_sided)
{
    normal = u.normalized().cross(v.normalized());
    area = u.length() * v.length();
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
        Vec3f point = (position + rnd[i][0]*u + rnd[i][1]*v);
        Vec3f dir = (point - surface.position);

        if(dir.dot(surface.normal) < EPSILON_F) // THIS IS INCORRECT IF WE HAVE A BACKFACING BXDF
            continue;

        float cos_theta = dir.dot(-normal);
        if(cos_theta < 0.f && !double_sided)
            continue;

        light_samples.emplace_back();
        LightSample &light_sample = light_samples.back();

        light_sample.position = point;
        light_sample.intensity = intensity;
        light_sample.pdf = dir.sqr_length() / (area * (fabs(cos_theta) + EPSILON_F));
    }

    return true;
}
