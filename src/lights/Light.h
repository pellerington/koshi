#pragma once

#include <base/Object.h>
#include <intersection/Intersect.h>
#include <Util/Attribute.h>

// TODO: Replace this with an "Evaluatable" attribute which is just name "light" ?
// TODO: Include light in the material like we have done with Volumes!
class Light : public Object
{
public:
    Light(const AttributeVec3f& intensity_attr) : intensity_attr(intensity_attr) {}

    Vec3f get_intensity(const float& u, const float& v, const float& w, const Intersect * intersect, Resources &resources)
    {
        return intensity_attr.get_value(u, v, w, resources);
    }
    
private:
    // Add saturation/spectrum here too.
    const AttributeVec3f intensity_attr;
};
