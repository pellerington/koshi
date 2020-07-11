#pragma once

#include <base/Object.h>
#include <embree/Embree.h>

// TODO: Find a better way to create these than attaching them to the Geometry? Can we just have table which converts GeometryType -> Function.
class EmbreeGeometry : public Object
{
public:
    const RTCGeometry& get_rtc_geometry() { return geometry; }
protected:
    RTCGeometry geometry;
};