#pragma once

#include <base/Object.h>
#include <embree/Embree.h>

class EmbreeGeometry : public Object
{
public:
    const RTCGeometry& get_rtc_geometry() { return geometry; }
protected:
    RTCGeometry geometry;
};