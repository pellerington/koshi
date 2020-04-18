#pragma once

#include <embree3/rtcore.h>
#include <intersection/Ray.h>

struct Embree
{
    static RTCDevice rtc_device;
};

struct EmbreeIntersectContext : public RTCIntersectContext
{
    Ray * ray = nullptr;
};