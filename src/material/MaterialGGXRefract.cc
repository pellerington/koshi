#include <material/MaterialGGXRefract.h>

#include <math/Helpers.h>
#include <Util/Color.h>
#include <cmath>
#include <iostream>
#include <integrator/AbsorbtionMedium.h>

MaterialGGXRefract::MaterialGGXRefract(const Texture * color_texture, const Texture * roughness_texture, 
                                       const float& ior, const float& color_depth, 
                                       const Texture * normal_texture, const Texture * opacity_texture)
: Material(normal_texture, opacity_texture), color_texture(color_texture), color_depth(color_depth), roughness_texture(roughness_texture), ior(ior)
{
}

MaterialLobes MaterialGGXRefract::instance(const Surface * surface, const Intersect * intersect, Resources& resources)
{
    MaterialLobes instance(resources.memory);
    MaterialLobeGGXRefract * lobe = resources.memory->create<MaterialLobeGGXRefract>();

    lobe->rng = resources.random_service->get_random<2>();

    lobe->surface = surface;
    lobe->wi = intersect->ray.dir;
    lobe->normal = normal_texture ? (intersect->geometry->get_obj_to_world() * normal_texture->evaluate<Vec3f>(surface->u, surface->v, surface->w, intersect, resources)).normalized() : surface->normal;
    lobe->normal = surface->facing ? lobe->normal : -lobe->normal;
    lobe->transform = Transform3f::basis_transform(lobe->normal);

    lobe->rng = resources.random_service->get_random<2>();
    lobe->ior_in = surface->facing ? 1.f : ior;
    lobe->ior_out = surface->facing ? ior : 1.f;
    lobe->color = color_texture->evaluate<Vec3f>(surface->u, surface->v, surface->w, intersect, resources);
    lobe->roughness = roughness_texture->evaluate<float>(surface->u, surface->v, surface->w, intersect, resources);
    lobe->roughness = clamp(lobe->roughness * lobe->roughness, 0.01f, 0.99f);
    lobe->roughness_sqr = lobe->roughness * lobe->roughness;
    lobe->fresnel = resources.memory->create<FresnelDielectric>(lobe->ior_in, lobe->ior_out);
    if(color_depth > EPSILON_F)
    {
        lobe->interior = surface->facing ? resources.memory->create<AbsorbtionMedium>(lobe->color, color_depth) : nullptr;
        lobe->color = VEC3F_ONES;
    }
    instance.push(lobe);
    return instance;
}

bool MaterialLobeGGXRefract::sample(MaterialSample& sample, Resources& resources) const
{
    const float * rnd = rng.rand();

    const float theta = TWO_PI * rnd[0];
    const float phi = atanf(roughness * sqrtf(rnd[1]) / sqrtf(1.f - rnd[1]));
    const Vec3f h = transform * Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta));

    const float h_dot_wi = clamp(h.dot(-wi), -1.f, 1.f);
    const float eta = ior_in / ior_out;
    const float k = 1.f - eta * eta * (1.f - h_dot_wi * h_dot_wi);
    if(k < 0.f) return false;

    sample.wo = eta * wi + (eta * fabs(h_dot_wi) - sqrtf(k)) * h;

    const float n_dot_h = clamp(h.dot(normal), -1.f, 1.f);
    const float h_dot_wo = clamp(h.dot(sample.wo), -1.f, 1.f);
    const float n_dot_wi = clamp(normal.dot(-wi), -1.f, 1.f);
    const float n_dot_wo = clamp(normal.dot(sample.wo), -1.f, 1.f);

    const float d = D(normal, h, n_dot_h, roughness_sqr);
    const float g = G1(-wi, normal, h, h_dot_wi, n_dot_wi, roughness_sqr)*G1(sample.wo, normal, h, h_dot_wo, n_dot_wo, roughness_sqr);
    const Vec3f f = fresnel->Ft(fabs(h_dot_wi));

    const float denom = std::pow(ior_in * h_dot_wi + ior_out * h_dot_wo + EPSILON_F, 2);

    sample.weight = ior_out * ior_out * (fabs(h_dot_wi) * fabs(h_dot_wo)) / (fabs(n_dot_wi) * fabs(n_dot_wo));
    sample.weight *= color * f * g * d * fabs(n_dot_wo) / denom;
    // sample.weight *= eta * eta;

    sample.pdf = d * n_dot_h * (ior_out * ior_out * fabs(h_dot_wo)) / denom;

    return sample.pdf > 0.f;
}

Vec3f MaterialLobeGGXRefract::weight(const Vec3f& wo, Resources& resources) const
{
    // TODO: evalualte this. Prehaps on the condition that we are exiting the object.
    return VEC3F_ZERO;

    if(wo.dot(wi) > 0.f)
        return VEC3F_ZERO;

    const Vec3f h = ((surface->facing) ? 1.f : -1.f) * (-wi*ior_in + wo*ior_out).normalized();

    const float n_dot_h = clamp(h.dot(normal), -1.f, 1.f);
    const float h_dot_wi = clamp(h.dot(-wi), -1.f, 1.f);
    const float h_dot_wo = clamp(h.dot(wo), -1.f, 1.f);
    const float n_dot_wi = clamp(normal.dot(-wi), -1.f, 1.f);
    const float n_dot_wo = clamp(normal.dot(wo), -1.f, 1.f);

    const float d = D(normal, h, n_dot_h, roughness_sqr);
    const float g = G1(-wi, normal, h, h_dot_wi, n_dot_wi, roughness_sqr)*G1(wo, normal, h, h_dot_wo, n_dot_wo, roughness_sqr);
    const Vec3f f = fresnel->Fr(fabs(h_dot_wi));

    const float denom = std::pow(ior_in * h_dot_wi + ior_out * h_dot_wo + EPSILON_F, 2);

    Vec3f weight = ior_out * ior_out * (fabs(h_dot_wi) * fabs(h_dot_wo)) / (fabs(n_dot_wi) * fabs(n_dot_wo));
    weight *= color * f * g * d * fabs(n_dot_wo) / denom;
    // sample.weight *= eta * eta;

    return weight;
}

float MaterialLobeGGXRefract::pdf(const Vec3f& wo, Resources& resources) const
{
    // TODO: evalualte this. Prehaps on the condition that we are exiting the object.
    return 0.f;

    if(wo.dot(wi) > 0.f)
        return 0.f;

    const Vec3f h = ((surface->facing) ? 1.f : -1.f) * (-wi*ior_in + wo*ior_out).normalized();

    const float n_dot_h = clamp(h.dot(normal), -1.f, 1.f);
    const float h_dot_wi = clamp(h.dot(-wi), -1.f, 1.f);
    const float h_dot_wo = clamp(h.dot(wo), -1.f, 1.f);

    const float d = D(normal, h, n_dot_h, roughness_sqr);

    const float denom = std::pow(ior_in * h_dot_wi + ior_out * h_dot_wo + EPSILON_F, 2);
    return d * n_dot_h * (ior_out * ior_out * fabs(h_dot_wo)) / denom;
}
