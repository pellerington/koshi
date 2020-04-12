#pragma once

#include <embree3/rtcore.h>

#include <Util/Ray.h>
#include <Volume/VolumeStack.h>

struct IntersectContext : public RTCIntersectContext
{
    Ray * ray = nullptr;
    VolumeStack * volumes = nullptr;
};

class Embree
{
public:
    static RTCDevice rtc_device;
};
