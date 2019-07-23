#pragma once

#include "Material.h"

class GGX : public Material
{
public:
    GGX(Vec3f specular_color = Vec3f::Zero(), float roughness = 0.f, float ior = 1.f, Vec3f emission = Vec3f::Zero());
    const Vec3f get_emission() { return emission; }
    bool sample_material(const Surface &surface, std::deque<SrfSample> &srf_samples, float sample_reduction = 1.f);
    bool evaluate_material(const Surface &surface, SrfSample &srf_sample, float &pdf);

    inline float D(const Vec3f &n, const Vec3f &h);
    inline float G1(const Vec3f &v, const Vec3f &n, const Vec3f &h);
    inline Vec3f F(const Vec3f &wi, const Vec3f &h);

private:
    Vec3f specular_color;
    float roughness;
    float roughness_sqr;
    float roughness_sqrt;
    float ior;
    Vec3f emission;
};
