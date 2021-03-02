#pragma once

#include <koshi/material/Lobe.h>

#define BACK_LAMBERT_UNIFORM_SAMPLE false

KOSHI_OPEN_NAMESPACE

struct BackLambert : public Lobe
{
    DEVICE_FUNCTION BackLambert() : Lobe(BACK_LAMBERT, BACK) {}
    
    Vec3f color;
    Vec3f normal;
    Transform world_transform;

    DEVICE_FUNCTION bool sample(Sample& sample, const Intersect& intersect, const Vec3f& wi, Random& random) const
    {
        const float theta = two_pi * random.rand();
    #if BACK_LAMBERT_UNIFORM_SAMPLE
        const float phi = acosf(random.rand());
        sample.wo = world_transform * Vec3f(sinf(phi) * cosf(theta), sinf(phi) * sinf(theta), cosf(phi));
        sample.pdf = inv_two_pi;
    #else
        const float r2 = random.rand();
        const float r = sqrtf(r2), z = sqrtf(max(epsilon, 1.f - r2));
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

        #if BACK_LAMBERT_UNIFORM_SAMPLE
            sample.pdf = inv_two_pi;
        #else
            sample.pdf = fabs(n_dot_wo) * inv_pi;
        #endif

        return true;
    }
};

KOSHI_CLOSE_NAMESPACE