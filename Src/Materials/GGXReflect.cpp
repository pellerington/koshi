#include "GGXReflect.h"

#include "../Math/RNG.h"
#include "../Math/Helpers.h"
#include "../Util/Color.h"
#include <cmath>
#include <iostream>

GGXReflect::GGXReflect(const Vec3f &specular_color, const float &roughness, std::shared_ptr<Fresnel> _fresnel, const Vec3f &emission)
: specular_color(specular_color), roughness(clamp(roughness*roughness, 0.000001f, 0.999999f)), roughness_sqr(this->roughness * this->roughness), roughness_sqrt(sqrtf(this->roughness)), fresnel(_fresnel), emission(emission)
{
}

std::shared_ptr<Material> GGXReflect::instance(const Surface &surface)
{
    std::shared_ptr<GGXReflect> material(new GGXReflect(*this));
    if(!fresnel)
        material->fresnel = std::shared_ptr<Fresnel>(new FresnelMetalic(specular_color));
    return material;
}

bool GGXReflect::sample_material(const Surface &surface, std::deque<PathSample> &path_samples, const float sample_reduction)
{
    if(!surface.enter)
        return false;

    // Estimate the number of samples
    const uint num_samples = std::max(1.f, SAMPLES_PER_SA * roughness_sqrt * sample_reduction);
    const float quality = 1.f / num_samples;

    const Transform3f transform = Transform3f::normal_transform(surface.normal);

    std::vector<Vec2f> rnd;
    RNG::Rand2d(num_samples, rnd);

    for(uint i = 0; i < rnd.size(); i++)
    {
        path_samples.emplace_back();
        PathSample &path_sample = path_samples.back();
        path_sample.quality = quality;

        const float theta = TWO_PI * rnd[i][0];
        const float phi = atanf(roughness * sqrtf(rnd[i][1]) / sqrtf(1.f - rnd[i][1]));
        const Vec3f h = transform * Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta));

        const float h_dot_wi = clamp(h.dot(-surface.wi), -1.f, 1.f);
        path_sample.wo = (2.f * h_dot_wi * h + surface.wi);

        const float n_dot_h = clamp(h.dot(surface.normal), -1.f, 1.f);
        const float h_dot_wo = clamp(h.dot(path_sample.wo), -1.f, 1.f);
        const float n_dot_wi = clamp(surface.normal.dot(-surface.wi), -1.f, 1.f);
        const float n_dot_wo = clamp(surface.normal.dot(path_sample.wo), -1.f, 1.f);

        const float d = D(surface.normal, h, n_dot_h, roughness_sqr);
        const float g = G1(-surface.wi, surface.normal, h, h_dot_wi, n_dot_wi, roughness_sqr)*G1(path_sample.wo, surface.normal, h, h_dot_wo, n_dot_wo, roughness_sqr);
        const Vec3f f = fresnel->Fr(fabs(h_dot_wi));

        path_sample.fr = (n_dot_wo > 0.f) ? (specular_color * f * g * d) / (4.f * n_dot_wi) : VEC3F_ZERO;
        path_sample.pdf = (d * n_dot_h) / (4.f * h_dot_wo);

        if(!path_sample.pdf || is_black(path_sample.fr))
            path_samples.pop_back();
    }

    return true;
}

bool GGXReflect::evaluate_material(const Surface &surface, PathSample &path_sample, float &pdf)
{
    if(path_sample.wo.dot(surface.normal) < 0)
        return false;

    const Vec3f h = (path_sample.wo - surface.wi).normalized();
    const float n_dot_h = clamp(h.dot(surface.normal), -1.f, 1.f);
    const float h_dot_wi = clamp(h.dot(-surface.wi), -1.f, 1.f);
    const float h_dot_wo = clamp(h.dot(path_sample.wo), -1.f, 1.f);
    const float n_dot_wi = clamp(surface.normal.dot(-surface.wi), -1.f, 1.f);
    const float n_dot_wo = clamp(surface.normal.dot(path_sample.wo), -1.f, 1.f);

    const float d = D(surface.normal, h, n_dot_h, roughness_sqr);
    const float g = G1(-surface.wi, surface.normal, h, h_dot_wi, n_dot_wi, roughness_sqr)*G1(path_sample.wo, surface.normal, h, h_dot_wo, n_dot_wo, roughness_sqr);
    const Vec3f f = fresnel->Fr(fabs(h_dot_wi));

    path_sample.fr = (specular_color * f * g * d) / (4.f * n_dot_wi);
    pdf = (d * n_dot_h) / (4.f * h_dot_wo);

    return true;
}
