#pragma once

#include <base/Object.h>
#include <texture/Texture.h>
#include <intersection/Intersect.h>

// TODO: Replace this with an "Evaluatable" attribute which is just name "light" ?
// TODO: Include light in the material like we have done with Volumes!
class Light : public Object
{
public:
    Light(const Texture * intensity) : intensity(intensity) {}

    Vec3f get_intensity(const float& u, const float& v, const float& w, const Intersect * intersect, Resources &resources)
    {
        return intensity->evaluate<Vec3f>(u, v, w, intersect, resources);
    }
    
private:
    // Add saturation/spectrum here too.
    const Texture * intensity;
};
