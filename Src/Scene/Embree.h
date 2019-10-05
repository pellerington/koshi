#pragma once

#include <embree3/rtcore.h>

struct RTCVertex   { float x, y, z, r; };
struct RTCTriangle { uint v0, v1, v2; };
struct RTCQuad     { uint v0, v1, v2, v3; };

class Embree
{
public:
    static RTCDevice rtc_device;
};
