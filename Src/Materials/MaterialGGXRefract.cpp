#include "MaterialGGXRefract.h"

#include "../Math/RNG.h"
#include "../Math/Helpers.h"
#include "../Util/Color.h"
#include <cmath>
#include <iostream>

MaterialGGXRefract::MaterialGGXRefract(const Vec3f &refractive_color, const float &roughness, const float &ior, std::shared_ptr<Fresnel> _fresnel, const Vec3f &emission)
: refractive_color(refractive_color), roughness(clamp(roughness*roughness, 0.000001f, 0.999999f)), roughness_sqr(this->roughness * this->roughness), roughness_sqrt(sqrtf(this->roughness)), ior(ior), fresnel(_fresnel), emission(emission)
{
}

std::shared_ptr<Material> MaterialGGXRefract::instance(const Surface &surface)
{
    std::shared_ptr<MaterialGGXRefract> material(new MaterialGGXRefract(*this));
    if(!fresnel)
        material->fresnel = (surface.enter) ? std::shared_ptr<Fresnel>(new FresnelDielectric(1.f, ior)) : std::shared_ptr<Fresnel>(new FresnelNone);
    return material;
}

bool MaterialGGXRefract::sample_material(const Surface &surface, std::deque<MaterialSample> &samples, const float sample_reduction)
{
    // Estimate the number of samples
    const uint num_samples = std::max(1.f, SAMPLES_PER_SA * roughness_sqrt * sample_reduction);
    const float quality = 1.f / num_samples;

    std::vector<Vec2f> rnd;
    RNG::Rand2d(num_samples, rnd);

    for(uint i = 0; i < rnd.size(); i++)
    {
        const float ior_in = surface.enter ? 1.f : ior;
        const float ior_out = surface.enter ? ior : 1.f;

        const float theta = TWO_PI * rnd[i][0];
        const float phi = atanf(roughness * sqrtf(rnd[i][1]) / sqrtf(1.f - rnd[i][1]));
        const Vec3f h = ((surface.enter) ? 1.f : -1.f) * (surface.transform * Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta)));

        const float h_dot_wi = clamp(h.dot(-surface.wi), -1.f, 1.f);
        const float eta = ior_in / ior_out;
        const float k = 1.f - eta * eta * (1.f - h_dot_wi * h_dot_wi);
        if(k < 0.f) continue;

        samples.emplace_back();
        MaterialSample &sample = samples.back();
        sample.quality = quality;
        sample.wo = eta * surface.wi + (eta * fabs(h_dot_wi) - sqrtf(k)) * h;

        const Vec3f normal = ((surface.enter) ? surface.normal : -surface.normal);

        const float n_dot_h = clamp(h.dot(normal), -1.f, 1.f);
        const float h_dot_wo = clamp(h.dot(sample.wo), -1.f, 1.f);
        const float n_dot_wi = clamp(normal.dot(-surface.wi), -1.f, 1.f);
        const float n_dot_wo = clamp(normal.dot(sample.wo), -1.f, 1.f);

        const float d = D(normal, h, n_dot_h, roughness_sqr);
        const float g = G1(-surface.wi, normal, h, h_dot_wi, n_dot_wi, roughness_sqr)*G1(sample.wo, normal, h, h_dot_wo, n_dot_wo, roughness_sqr);
        const Vec3f f = fresnel->Ft(fabs(h_dot_wi));

        const float denom = std::pow(ior_in * h_dot_wi + ior_out * h_dot_wo, 2);

        sample.fr = ior_out * ior_out * (fabs(h_dot_wi) * fabs(h_dot_wo)) / (fabs(n_dot_wi) * fabs(n_dot_wo));
        sample.fr *= refractive_color * f * g * d * fabs(n_dot_wo) / denom;
        // sample.fr *= eta * eta;

        sample.pdf = d * n_dot_h * (ior_out * ior_out * fabs(h_dot_wo)) / denom;

        if(!sample.pdf || is_black(sample.fr))
            samples.pop_back();
    }

    return true;
}

bool MaterialGGXRefract::evaluate_material(const Surface &surface, MaterialSample &sample)
{
    if(sample.wo.dot(surface.wi) > 0.f)
        return false;

    const float ior_in = surface.enter ? 1.f : ior;
    const float ior_out = surface.enter ? ior : 1.f;

    const Vec3f normal = ((surface.enter) ? surface.normal : -surface.normal);
    const Vec3f h = ((surface.enter) ? 1.f : -1.f) * (-surface.wi*ior_in + sample.wo*ior_out).normalized();

    const float n_dot_h = clamp(h.dot(normal), -1.f, 1.f);
    const float h_dot_wi = clamp(h.dot(-surface.wi), -1.f, 1.f);
    const float h_dot_wo = clamp(h.dot(sample.wo), -1.f, 1.f);
    const float n_dot_wi = clamp(normal.dot(-surface.wi), -1.f, 1.f);
    const float n_dot_wo = clamp(normal.dot(sample.wo), -1.f, 1.f);

    const float d = D(normal, h, n_dot_h, roughness_sqr);
    const float g = G1(-surface.wi, normal, h, h_dot_wi, n_dot_wi, roughness_sqr)*G1(sample.wo, normal, h, h_dot_wo, n_dot_wo, roughness_sqr);
    const Vec3f f = fresnel->Fr(fabs(h_dot_wi));

    const float denom = std::pow(ior_in * h_dot_wi + ior_out * h_dot_wo, 2);

    sample.fr = ior_out * ior_out * (fabs(h_dot_wi) * fabs(h_dot_wo)) / (fabs(n_dot_wi) * fabs(n_dot_wo));
    sample.fr *= refractive_color * f * g * d * fabs(n_dot_wo) / denom;
    // sample.fr *= eta * eta;

    sample.pdf = d * n_dot_h * (ior_out * ior_out * fabs(h_dot_wo)) / denom;

    return true;
}
