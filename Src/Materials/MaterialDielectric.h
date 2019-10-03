#pragma once

#include "Material.h"
#include "Fresnel.h"
#include "MaterialGGXReflect.h"
#include "MaterialGGXRefract.h"

class MaterialDielectric : public Material
{
public:
    MaterialDielectric(const Vec3f &reflective_color = VEC3F_ZERO, const Vec3f &refractive_color = VEC3F_ZERO,
               const float &roughness = 0.f, const float &ior = 1.f, const Vec3f &emission = VEC3F_ZERO);
    std::shared_ptr<Material> instance(const Surface &surface);

    Type get_type() { return Material::Dielectric; }

    bool sample_material(const Surface &surface, std::deque<MaterialSample> &samples, float sample_reduction = 1.f);
    bool evaluate_material(const Surface &surface, MaterialSample &sample);
    const Vec3f get_emission() { return emission; }

private:
    std::shared_ptr<Fresnel> fresnel;
    const float ior;
    const Vec3f emission;
    std::shared_ptr<MaterialGGXReflect> ggx_reflect;
    std::shared_ptr<MaterialGGXRefract> ggx_refract;
};
