#pragma once

#include "../Math/Vec3f.h"
#include <vector>

// Move this to its own place so we can give it it's own constructor
class VolumeProperties
{
public:
    VolumeProperties(const Vec3f &max_density) : max_density(max_density) {}
    // VolumeProperties(const std::shared_ptr<MaterialVolume> &material) : material(material) {}
    // std::shared_ptr<MaterialVolume> material;
    Vec3f max_density, min_density;
    // bool is_homogenous?
    // bool is_multiscat?
};

struct Volume
{
    Vec3f max_density, min_density;
    float tmin, tmax;
    std::vector<VolumeProperties*> volume_prop;
    //Vector<Vec3f> UVW_BEGIN, UVW_LEN

    //Integration Type Or should that be on the stack itself??? Or decided by the Integrator?
};
