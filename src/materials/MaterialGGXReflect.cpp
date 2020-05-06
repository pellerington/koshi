#include <materials/MaterialGGXReflect.h>

#include <Math/Helpers.h>
#include <Util/Color.h>
#include <cmath>
#include <iostream>

MaterialGGXReflect::MaterialGGXReflect(const AttributeVec3f &specular_color_attribute, const AttributeFloat &roughness_attribute)
: specular_color_attribute(specular_color_attribute), roughness_attribute(roughness_attribute)
{
}

MaterialInstance MaterialGGXReflect::instance(const GeometrySurface * surface, Resources &resources)
{
    MaterialInstance instance;
    MaterialLobeGGXReflect * lobe = resources.memory.create<MaterialLobeGGXReflect>();
    lobe->surface = surface;
    lobe->rng = resources.random_number_service.get_random_2D();
    lobe->color = specular_color_attribute.get_value(surface->u, surface->v, 0.f, resources);
    lobe->roughness = roughness_attribute.get_value(surface->u, surface->v, 0.f, resources);
    lobe->roughness = clamp(lobe->roughness * lobe->roughness, 0.01f, 0.99f);
    lobe->roughness_sqr = lobe->roughness * lobe->roughness;
    lobe->fresnel = resources.memory.create<FresnelMetalic>(lobe->color);
    instance.push(lobe);
    return instance;
}

bool MaterialLobeGGXReflect::sample(MaterialSample& sample, Resources& resources) const
{
    const Vec2f rnd = rng.rand();

    const float theta = TWO_PI * rnd[0];
    const float phi = atanf(roughness * sqrtf(rnd[1]) / sqrtf(1.f - rnd[1]));
    const Vec3f h = (surface->facing ? 1.f : -1.f) * (surface->transform * Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta)));
    const float h_dot_wi = clamp(h.dot(-surface->wi), -1.f, 1.f);

    // If we are inside the only time we want to call this is if we have total internal reflection.
    const Vec3f f = fresnel->Fr(fabs(h_dot_wi));
    if(!surface->facing && f < 1.f)
        return false;

    sample.wo = (2.f * h_dot_wi * h + surface->wi);

    const Vec3f normal = (surface->facing ? 1.f : -1.f) * surface->normal;
    const float n_dot_h = clamp(h.dot(normal), -1.f, 1.f);
    const float h_dot_wo = clamp(h.dot(sample.wo), -1.f, 1.f);
    const float n_dot_wi = fabs(surface->n_dot_wi);
    const float n_dot_wo = clamp(normal.dot(sample.wo), -1.f, 1.f);

    const float d = D(normal, h, n_dot_h, roughness_sqr);
    const float g = G1(-surface->wi, normal, h, h_dot_wi, n_dot_wi, roughness_sqr)*G1(sample.wo, normal, h, h_dot_wo, n_dot_wo, roughness_sqr);

    sample.weight = (n_dot_wo > 0.f) ? (color * f * g * d) / (4.f * n_dot_wi) : VEC3F_ZERO;
    sample.pdf = (d * n_dot_h) / (4.f * h_dot_wo);

    return sample.pdf > 0.01f;
}

bool MaterialLobeGGXReflect::evaluate(MaterialSample& sample, Resources& resources) const
{
    if(sample.wo.dot(surface->normal) < 0)
        return false;

    const Vec3f h = (sample.wo - surface->wi).normalized();
    const float n_dot_h = clamp(h.dot(surface->normal), -1.f, 1.f);
    const float h_dot_wi = clamp(h.dot(-surface->wi), -1.f, 1.f);
    const float h_dot_wo = clamp(h.dot(sample.wo), -1.f, 1.f);
    const float n_dot_wi = surface->n_dot_wi;
    const float n_dot_wo = clamp(surface->normal.dot(sample.wo), -1.f, 1.f);

    const float d = D(surface->normal, h, n_dot_h, roughness_sqr);
    const float g = G1(-surface->wi, surface->normal, h, h_dot_wi, n_dot_wi, roughness_sqr)*G1(sample.wo, surface->normal, h, h_dot_wo, n_dot_wo, roughness_sqr);
    const Vec3f f = fresnel->Fr(fabs(h_dot_wi));

    sample.weight = (color * f * g * d) / (4.f * n_dot_wi);
    sample.pdf = (d * n_dot_h) / (4.f * h_dot_wo);

    return true;
}
