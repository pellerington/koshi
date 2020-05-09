#pragma once

#include <base/Object.h>
#include <intersection/Intersect.h>
#include <Util/Attribute.h>

class Light : public Object
{
public:
    Light(const AttributeVec3f& intensity_attr) : intensity_attr(intensity_attr) {}

    Vec3f get_intensity(const Intersect * intersect, Resources &resources)
    {
        return intensity_attr.get_value(intersect->surface.u, intersect->surface.v, 0.f, resources);;
    }
    
private:
    // Add saturation/spectrum here too.
    const AttributeVec3f intensity_attr;
};
