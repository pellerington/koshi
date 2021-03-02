#pragma once

#include <koshi/material/Lambert.h>
#include <koshi/material/BackLambert.h>
#include <koshi/material/Reflect.h>

KOSHI_OPEN_NAMESPACE

DEVICE_FUNCTION bool sample_lobe(const Lobe * lobe, Sample& sample, const Intersect& intersect, const Ray& ray, Random& random)
{
    switch(lobe->getType())
    {
        case Lobe::LAMBERT:
            return ((const Lambert *)lobe)->sample(sample, intersect, ray.direction, random);
        case Lobe::BACK_LAMBERT:
            return ((const BackLambert *)lobe)->sample(sample, intersect, ray.direction, random);
        case Lobe::REFLECT:
            return ((const Reflect *)lobe)->sample(sample, intersect, ray.direction);
        default:
            return false;
    }
}

DEVICE_FUNCTION bool evaluate_lobe(const Lobe * lobe, Sample& sample, const Intersect& intersect, const Ray& ray)
{
    switch(lobe->getType())
    {
        case Lobe::LAMBERT:
            return ((const Lambert *)lobe)->evaluate(sample, intersect, ray.direction);
        case Lobe::BACK_LAMBERT:
            return ((const BackLambert *)lobe)->evaluate(sample, intersect, ray.direction);
        case Lobe::REFLECT:
            return ((const Reflect *)lobe)->evaluate(sample, intersect, ray.direction);
        default:
            return false;
    }
}

KOSHI_CLOSE_NAMESPACE