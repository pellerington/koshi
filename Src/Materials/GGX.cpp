#include "GGX.h"

#include "../Math/RNG.h"
#include "../Math/Helpers.h"
#include "../Util/Color.h"
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>

#define UNIFORM_SAMPLE 0

GGX::GGX(Vec3f specular_color, float roughness, float ior, Vec3f emission)
: specular_color(specular_color), roughness(clamp(roughness, 0.000001f, 0.999999f)), roughness_sqr(this->roughness * this->roughness), roughness_sqrt(sqrtf(this->roughness)), ior(ior), emission(emission)
{
}

bool GGX::sample_material(const Surface &surface, std::deque<SrfSample> &srf_samples, float sample_reduction)
{
    // Estimate the number of samples
    const uint num_samples = std::max(1.f, SAMPLES_PER_SA * roughness_sqrt * sample_reduction);

    Eigen::Matrix3f transform = world_transform(surface.normal);

    std::vector<Vec2f> rnd;
    RNG::StratifiedRand(num_samples, rnd);

    for(uint i = 0; i < num_samples; i++)
    {
        srf_samples.emplace_back();
        SrfSample &srf_sample = srf_samples.back();
        srf_sample.type = SrfSample::Material;

#if UNIFORM_SAMPLE
        // Uniform Sample
        float theta = TWO_PI * rnd[i][0];
        float phi = acosf(rnd[i][1]);
        srf_sample.wo = transform * Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta));

        Vec3f h = (srf_sample.wo - surface.wi).normalized();
        float wi_dot_n = clamp(surface.wi.dot(-surface.normal), 0.f, 1.f);//h or n?
        float wo_dot_n = clamp(srf_sample.wo.dot(surface.normal), 0.f, 1.f); // HOW ARE WE GETTING Wo = Wi??!!
        float d = D(surface.normal, h);
        float g = G1(-surface.wi, surface.normal, h)*G1(srf_sample.wo, surface.normal, h);
        Vec3f f = F(surface.wi, h);

        srf_sample.fr = (wo_dot_n) ? (specular_color * f * g * d) / (4.f * wo_dot_n * wi_dot_n) : Vec3f::Zero();
        srf_sample.pdf = INV_TWO_PI;
#else
        float theta = TWO_PI * rnd[i][0];
        float phi = atanf(roughness * sqrtf(rnd[i][1]) / sqrtf(1.f - rnd[i][1]));
        Vec3f h = transform * Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta));
        srf_sample.wo = (2.f * h.dot(-surface.wi) * h + surface.wi);

        float n_dot_h = clamp(surface.normal.dot(h), 0.f, 1.f);
        float wi_dot_n = clamp(surface.wi.dot(-surface.normal), 0.f, 1.f);//h or n?
        float wo_dot_n = clamp(srf_sample.wo.dot(surface.normal), 0.f, 1.f); // HOW ARE WE GETTING Wo = Wi??!!
        float d = D(surface.normal, h);
        float g = G1(-surface.wi, surface.normal, h)*G1(srf_sample.wo, surface.normal, h);
        Vec3f f = F(surface.wi, h);

        srf_sample.fr = (wo_dot_n) ? (specular_color * f * g * d) / (4.f * wo_dot_n * wi_dot_n) : Vec3f::Zero();
        srf_sample.pdf = (d * n_dot_h) / (4.f * srf_sample.wo.dot(h));
#endif

        if(!srf_sample.pdf || is_black(srf_sample.fr))
            srf_samples.pop_back();
    }

    return true;
}

bool GGX::evaluate_material(const Surface &surface, SrfSample &srf_sample, float &pdf)
{
    if(srf_sample.wo.dot(surface.normal) < 0)
        return false;

    Vec3f h = (srf_sample.wo - surface.wi).normalized();
    float wi_dot_n = clamp(surface.wi.dot(-surface.normal), 0.000001f, 1.f);//h or n?
    float wo_dot_n = clamp(srf_sample.wo.dot(surface.normal), 0.000001f , 1.f);

    float d = D(surface.normal, h);
    float g = G1(-surface.wi, surface.normal, h)*G1(srf_sample.wo, surface.normal, h);
    Vec3f f = F(surface.wi, h);

    // BRDF
    srf_sample.fr = (specular_color.cwiseProduct(f) * g * d) / (4.f * wo_dot_n * wi_dot_n);


#if UNIFORM_SAMPLE
        pdf = INV_TWO_PI;
#else
        float n_dot_h = clamp(surface.normal.dot(h), 0.f, 1.f);
        pdf = (d * n_dot_h) / (4.f * srf_sample.wo.dot(h));
#endif

    return true;
}

inline float GGX::D(const Vec3f &n, const Vec3f &h)
{
    //Distribution GGX ( https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf )
    float n_dot_h = clamp(n.dot(h), 0.f, 1.f); // Should be passed in
    float tan_h = n.cross(h).norm() / n_dot_h;
    float det = roughness_sqr + (tan_h * tan_h);
    return (((n_dot_h > 0) ? 1.f : 0.f) * roughness_sqr) / (PI * std::pow(n_dot_h, 4) * det * det);
}

inline float GGX::G1(const Vec3f &v, const Vec3f &n, const Vec3f &h)
{
    //Geometric Term GGX (This G is actually G1? Check all terms in PAPER asap) ( https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf )
    float v_dot_h = clamp(v.dot(h), 0.f, 1.f);
    float v_dot_n = clamp(v.dot(n), 0.f, 1.f); // Should this be h or n?
    float x = ((v_dot_h / v_dot_n) > 0.f) ? 1.f : 0.f;
    // float v_dot_h_sqr = v_dot_h * v_dot_h; // add this back in?
    float tan_sqr = std::pow(v.cross(n).norm() / v.dot(n), 2);//(1.f - v_dot_h_sqr) / v_dot_h_sqr;

    return (x * 2.f) / (1.f + sqrtf(1.f + roughness_sqr * tan_sqr));
}

inline Vec3f GGX::F(const Vec3f &wi, const Vec3f &h)
{
    //Fresnal ( Inaccurate look for good solution later  https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf)
    float wi_dot_h = clamp(wi.dot(-h), 0.f, 1.f);
    float curr_ior = 1.f;
    Vec3f F0 = std::abs((ior - curr_ior) / (ior + curr_ior)) * Vec3f::Ones();
    F0 = F0.cwiseProduct(F0);
    F0 = specular_color; // LERP BETWEEN F0 AND SPEC COLOR FOR METALLIC ATTRIBUTE
    return F0 + (Vec3f::Ones() - F0) * std::pow(1.f - wi_dot_h, 5);
}
