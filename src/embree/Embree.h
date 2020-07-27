#pragma once

#include <embree3/rtcore.h>
#include <koshi/intersection/Ray.h>
#include <koshi/intersection/Intersect.h>

struct Embree
{
    static RTCDevice rtc_device;

    // Static Helpers.
    static Vec3f normal(const RTCFilterFunctionNArguments * args) {
        return Vec3f(RTCHitN_Ng_x(args->hit, args->N, 0), 
                     RTCHitN_Ng_y(args->hit, args->N, 0), 
                     RTCHitN_Ng_z(args->hit, args->N, 0));
    }
};

struct EmbreeIntersectContext : public RTCIntersectContext
{
    IntersectList * intersects;
    Resources * resources;
};