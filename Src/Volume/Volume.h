#pragma once

#include "../Math/Vec3f.h"
#include "../Materials/Material.h"
#include "../Math/RNG.h"
class Volume;

struct VolumeSample
{
    Vec3f pos;
    Vec3f wo;
    Vec3f weight;
    // float pdf;
    float quality = 1.f;
    const std::vector<Volume*> * exit_volumes;
};

class Volume
{
public:
    Volume(const float &density = 0.f, const Vec3f &scattering = VEC3F_ZERO, const float &g = 0.f, const Vec3f &transparency = VEC3F_ONES, const Vec3f &emission = VEC3F_ZERO);

    Vec3f max_density, min_density;

    virtual bool is_heterogeneous() { return false; }
    // is_multiscattering???
    // sample_lights???

    virtual bool sample_volume(const Vec3f &wi, VolumeSample &sample); // UVW as well?
    virtual bool evaluate_volume(const Vec3f &wi, VolumeSample &sample); // UVW as well?

    //virtual const float anistropy() { return -1 < x < 1 } // Probably not needed
    virtual Vec3f get_density(const Vec3f &uvw = VEC3F_ZERO);
    virtual Vec3f get_scattering(const Vec3f &uvw = VEC3F_ZERO);
    virtual Vec3f get_emission(const Vec3f &uvw = VEC3F_ZERO);

private:
    const Vec3f density;
    const Vec3f scattering;
    const Vec3f emission;

    const float g;
    const float g_sqr;
    const float g_inv;
    const float g_abs;

};
