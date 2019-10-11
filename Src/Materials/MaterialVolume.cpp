#pragma once

#include "MaterialVolume.h"

MaterialVolume::MaterialVolume(const Vec3f &density, const Vec3f &scattering = VEC3F_ZERO, const Vec3f &emission = VEC3F_ZERO)
{

}

std::shared_ptr<Material> MaterialVolume::instance(const Surface * surface = nullptr)
{
    // return std::shared_ptr<Material>(this);
    return nullptr;
}

bool MaterialVolume::sample_material(std::vector<MaterialSample> &samples, float sample_reduction = 1.f)
{
    return false;
}
bool MaterialVolume::evaluate_material(MaterialSample &sample)
{
    return false;
}

const Vec3f MaterialVolume::get_density()
{
    return VEC3F_ZERO;
}
const Vec3f MaterialVolume::get_absorbtion()
{
    return VEC3F_ZERO;
}
const Vec3f MaterialVolume::get_scattering()
{
    return VEC3F_ZERO;
}
const Vec3f MaterialVolume::get_emission()
{
    return VEC3F_ZERO;
}
