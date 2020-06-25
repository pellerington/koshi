#include <material/MaterialGGXReflect.h>

#include <Math/Helpers.h>
#include <Util/Color.h>
#include <cmath>
#include <iostream>

MaterialGGXReflect::MaterialGGXReflect(const Texture * color_texture, const Texture * roughness_texture)
: color_texture(color_texture), roughness_texture(roughness_texture)
{
}

MaterialInstance MaterialGGXReflect::instance(const Surface * surface, const Intersect * intersect, Resources& resources)
{
    MaterialInstance instance(resources.memory);
    MaterialLobeGGXReflect * lobe = resources.memory->create<MaterialLobeGGXReflect>();

    lobe->rng = resources.random_service->get_random_2D();

    lobe->surface = surface;
    lobe->wi = intersect->ray.dir;
    lobe->normal = ((surface->facing) ? surface->normal : -surface->normal);
    lobe->transform = Transform3f::basis_transform(lobe->normal);

    lobe->color = color_texture->evaluate<Vec3f>(surface->u, surface->v, surface->w, intersect, resources);
    lobe->roughness = roughness_texture->evaluate<float>(surface->u, surface->v, surface->w, intersect, resources);
    lobe->roughness = clamp(lobe->roughness * lobe->roughness, 0.01f, 0.99f);
    lobe->roughness_sqr = lobe->roughness * lobe->roughness;
    lobe->fresnel = resources.memory->create<FresnelMetalic>(lobe->color);

    instance.push(lobe);
    return instance;
}

bool MaterialLobeGGXReflect::sample(MaterialSample& sample, Resources& resources) const
{
    const Vec2f rnd = rng.rand();

    const float theta = TWO_PI * rnd[0];
    const float phi = atanf(roughness * sqrtf(rnd[1]) / sqrtf(1.f - rnd[1]));
    const Vec3f h = transform * Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta));
    const float h_dot_wi = clamp(h.dot(-wi), -1.f, 1.f);

    // If we are inside the only time we want to call this is if we have total internal reflection.
    const Vec3f f = fresnel->Fr(fabs(h_dot_wi));
    if(!surface->facing && f < 1.f)
        return false;

    sample.wo = (2.f * h_dot_wi * h + wi);

    const float n_dot_h = clamp(h.dot(normal), -1.f, 1.f);
    const float h_dot_wo = clamp(h.dot(sample.wo), -1.f, 1.f);
    const float n_dot_wi = clamp(normal.dot(-wi), -1.f, 1.f);
    const float n_dot_wo = clamp(normal.dot(sample.wo), -1.f, 1.f);

    const float d = D(normal, h, n_dot_h, roughness_sqr);
    const float g = G1(-wi, normal, h, h_dot_wi, n_dot_wi, roughness_sqr)*G1(sample.wo, normal, h, h_dot_wo, n_dot_wo, roughness_sqr);

    sample.weight = (n_dot_wo > 0.f) ? (color * f * g * d) / (4.f * n_dot_wi) : VEC3F_ZERO;
    sample.pdf = (d * n_dot_h) / (4.f * h_dot_wo);

    return sample.pdf > 0.01f;
}

Vec3f MaterialLobeGGXReflect::weight(const Vec3f& wo, Resources& resources) const
{
    if(wo.dot(normal) < 0.f)
        return VEC3F_ZERO;

    const Vec3f h = (wo - wi).normalized();
    const float n_dot_h = clamp(h.dot(normal), -1.f, 1.f);
    const float h_dot_wi = clamp(h.dot(-wi), -1.f, 1.f);
    const float h_dot_wo = clamp(h.dot(wo), -1.f, 1.f);
    const float n_dot_wi = clamp(normal.dot(-wi), -1.f, 1.f);
    const float n_dot_wo = clamp(normal.dot(wo), -1.f, 1.f);

    const float d = D(normal, h, n_dot_h, roughness_sqr);
    const float g = G1(-wi, normal, h, h_dot_wi, n_dot_wi, roughness_sqr)*G1(wo, normal, h, h_dot_wo, n_dot_wo, roughness_sqr);
    const Vec3f f = fresnel->Fr(fabs(h_dot_wi));

    return (color * f * g * d) / (4.f * n_dot_wi);
}

float MaterialLobeGGXReflect::pdf(const Vec3f& wo, Resources& resources) const
{
    if(wo.dot(normal) < 0.f)
        return 0.f;

    const Vec3f h = (wo - wi).normalized();
    const float n_dot_h = clamp(h.dot(normal), -1.f, 1.f);
    const float h_dot_wo = clamp(h.dot(wo), -1.f, 1.f);

    const float d = D(normal, h, n_dot_h, roughness_sqr);

    return (d * n_dot_h) / (4.f * h_dot_wo);
}
