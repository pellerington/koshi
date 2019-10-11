#pragma once

#include "../Materials/MaterialVolume.h"
#include "../Math/Vec3f.h"
#include <vector>

// Move this to its own place so we can give it it's own constructor
class Volume
{
public:
    Volume(const Vec3f &max_density) : max_density(max_density) {}
    // Volume(const std::shared_ptr<MaterialVolume> &material) : material(material) {}
    std::shared_ptr<MaterialVolume> material;
    Vec3f max_density, min_density;
    // bool is_homogenous?
    // bool is_multiscat?
};
