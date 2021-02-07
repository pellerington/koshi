#pragma once

#include <koshi/material/Lobe.h>

#define UNIFORM_SAMPLE false

KOSHI_OPEN_NAMESPACE

struct BackLambert : public Lobe
{
    DEVICE_FUNCTION BackLambert() : Lobe(BACK_LAMBERT, BACK) {}
    Vec3f color;
    Vec3f normal;
    Transform world_transform;

    DEVICE_FUNCTION bool sample(Sample& sample, const Vec2f& rnd, const Intersect& intersect, const Vec3f& wi) const
    {
        const float theta = two_pi * rnd[0];
    #if UNIFORM_SAMPLE
        const float phi = acosf(rnd[1]);
        sample.wo = world_transform * Vec3f(sinf(phi) * cosf(theta), sinf(phi) * sinf(theta), cosf(phi));
        sample.pdf = inv_two_pi;
    #else
        const float r = sqrtf(rnd[1]), z = sqrtf(max(epsilon, 1.f - rnd[1]));
        sample.wo = world_transform * Vec3f(r * cosf(theta), r * sinf(theta), z);
        sample.pdf = z * inv_pi;
    #endif
        sample.value = color * inv_pi * sample.wo.dot(normal);

        if(intersect.facing) 
            sample.wo = -sample.wo; 

        return true;
    }

    DEVICE_FUNCTION bool evaluate(Sample& sample, const Intersect& intersect, const Vec3f& wi) const
    {
        const float n_dot_wo = normal.dot(sample.wo);
        const float n_dot_wi = normal.dot(wi);

        if(n_dot_wo * n_dot_wi < 0.f)
            return false;

        sample.value = color * inv_pi * fabs(n_dot_wo);

        #if UNIFORM_SAMPLE
            sample.pdf = inv_two_pi;
        #else
            sample.pdf = fabs(n_dot_wo) * inv_pi;
        #endif

        return true;
    }
};

KOSHI_CLOSE_NAMESPACE