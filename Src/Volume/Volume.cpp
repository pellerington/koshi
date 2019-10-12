#include "Volume.h"

Volume::Volume(const float &_density, const Vec3f &_scattering, const Vec3f &_transparency, const Vec3f &_emission)
: density(_transparency*_density), scattering(_scattering), emission(_emission)
{

    // if(!is_heterogeneous())
    // {
        max_density = get_density();
    // }

}

bool Volume::sample_volume(const Vec3f &wi, std::vector<MaterialSample> &samples, float sample_reduction)
{
    return false;
}

bool Volume::evaluate_volume(const Vec3f &wi, MaterialSample &sample)
{
    return false;
}

Vec3f Volume::get_density(const Vec3f &uvw)
{
    return density;
}
Vec3f Volume::get_scattering(const Vec3f &uvw)
{
    return scattering;
}
Vec3f Volume::get_emission(const Vec3f &uvw)
{
    return emission;
}
