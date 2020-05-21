#pragma once

#include <embree3/rtcore.h>
#include <intersection/Ray.h>
#include <intersection/Intersect.h>

struct Embree
{
    static RTCDevice rtc_device;
};

struct EmbreeIntersectContext : public RTCIntersectContext
{
    IntersectList * intersects;
    Integrator * default_integrator;
    Resources * resources;
};