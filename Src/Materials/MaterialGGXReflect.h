#pragma once

#include "Material.h"
#include "Fresnel.h"
#include "GGX.h"

class MaterialGGXReflect : public Material
{
public:
    MaterialGGXReflect(const Vec3f &specular_color = VEC3F_ZERO, const float &roughness = 0.f, std::shared_ptr<Fresnel> fresnel = nullptr, const Vec3f &emission = VEC3F_ZERO);
    std::shared_ptr<Material> instance(const Surface &surface);

    Type get_type() { return Material::GGXReflect; }

    bool sample_material(const Surface &surface, std::deque<MaterialSample> &samples, float sample_reduction = 1.f);
    bool evaluate_material(const Surface &surface, MaterialSample &sample);
    const Vec3f get_emission() { return emission; }
    void set_fresnel(std::shared_ptr<Fresnel> _fresnel) { fresnel = _fresnel; }

private:
    const Vec3f specular_color;
    const float roughness;
    const float roughness_sqr;
    const float roughness_sqrt;
    std::shared_ptr<Fresnel> fresnel;
    const Vec3f emission;
};
