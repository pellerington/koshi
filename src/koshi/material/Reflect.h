#pragma once

#include <koshi/material/Lobe.h>
#include <koshi/math/Constants.h>

KOSHI_OPEN_NAMESPACE

struct Reflect : public Lobe
{
    DEVICE_FUNCTION Reflect() : Lobe(REFLECT, FRONT) {}
    
    Vec3f color;
    Vec3f normal;

    DEVICE_FUNCTION bool sample(Sample& sample, const Intersect& intersect, const Vec3f& wi) const
    {
        sample.wo = wi - 2.f * wi.dot(normal) * normal; 
        sample.value = color * inv_epsilon * sample.wo.dot(normal);
        sample.pdf = inv_epsilon;
        return true;
    }

    DEVICE_FUNCTION bool evaluate(Sample& sample, const Intersect& intersect, const Vec3f& wi) const
    {
        return true;
    }
};

KOSHI_CLOSE_NAMESPACE