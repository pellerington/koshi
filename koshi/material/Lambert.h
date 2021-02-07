#pragma once

#include <koshi/material/Lobe.h>
#include <koshi/math/Constants.h>

#define LAMBERT_UNIFORM_SAMPLE false

KOSHI_OPEN_NAMESPACE

struct Lambert : public Lobe
{
    DEVICE_FUNCTION Lambert() : Lobe(LAMBERT, FRONT) {}
    Vec3f color;
    Vec3f normal;
    Transform world_transform;

    DEVICE_FUNCTION bool sample(Sample& sample, const Vec2f& rnd, const Intersect& intersect, const Vec3f& wi) const
    {
        // Only reflect lambert on the front of materials
        if(!intersect.facing)
            return false;

        const float theta = two_pi * rnd[0];
    #if LAMBERT_UNIFORM_SAMPLE
        const float phi = acosf(rnd[1]);
        sample.wo = world_transform * Vec3f(sinf(phi) * cosf(theta), sinf(phi) * sinf(theta), cosf(phi));
        sample.pdf = inv_two_pi;
    #else
        const float r = sqrtf(rnd[1]), z = sqrtf(max(epsilon, 1.f - rnd[1]));
        sample.wo = world_transform * Vec3f(r * cosf(theta), r * sinf(theta), z);
        sample.pdf = z * inv_pi;
    #endif
        sample.value = color * inv_pi * sample.wo.dot(normal);

        return true;
    }

    DEVICE_FUNCTION bool evaluate(Sample& sample, const Intersect& intersect, const Vec3f& wi) const
    {
        if(!intersect.facing)
            return false;

        const float n_dot_wo = normal.dot(sample.wo);

        if(n_dot_wo > 0.f)
            return false;

        sample.value = color * inv_pi * fabs(n_dot_wo);

        #if LAMBERT_UNIFORM_SAMPLE
            sample.pdf = inv_two_pi;
        #else
            sample.pdf = fabs(n_dot_wo) * inv_pi;
        #endif

        return true;
    }
};

KOSHI_CLOSE_NAMESPACE