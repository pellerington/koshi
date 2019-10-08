#pragma once

#include "../Math/Vec3f.h"

struct VolumeProperties
{
    VolumeProperties(const Vec3f &density) : density(density) {}
    Vec3f density;
};

struct Volume
{
    Vec3f density;
    float tmin, tmax;
    //UVW start end
    //std::shared_ptr<MaterialVolume>;
    //float max_density, min_density?;
    //Integration Type?
};
