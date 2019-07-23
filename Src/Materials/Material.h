#pragma once

#include <vector>
#include <queue>
#include "../Math/Types.h"
#include "../Util/Surface.h"
class SrfSample;
class Surface;

class Material
{
public:
    virtual const Vec3f get_emission() = 0;
    virtual bool sample_material(const Surface &surface, std::deque<SrfSample> &srf_samples, float sample_reduction = 1.f) = 0;
    virtual bool evaluate_material(const Surface &surface, SrfSample &srf_sample, float &pdf) = 0;
};
