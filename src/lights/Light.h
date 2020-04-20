#pragma once

#include <Util/Attribute.h>

class Light
{
public:
    Light(const AttributeVec3f& intensity_attr) : intensity_attr(intensity_attr) {}

    Vec3f get_intensity(const Surface& surface, Resources &resources)
    {
        return intensity_attr.get_value(surface.u, surface.v, 0.f, resources);;
    }
    
private:
    // Add saturation/spectrum here too.
    const AttributeVec3f intensity_attr;
};
