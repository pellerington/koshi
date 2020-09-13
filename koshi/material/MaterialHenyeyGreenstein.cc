#include <koshi/material/MaterialHenyeyGreenstein.h>

MaterialLobes MaterialHenyeyGreenstein::instance(const Surface * surface, const Intersect * intersect, Resources& resources)
{
    MaterialLobes instance(resources.memory);
    MaterialLobeHenyeyGreenstein * lobe = resources.memory->create<MaterialLobeHenyeyGreenstein>();
    lobe->rng = resources.random_service->get_random<2>();
    lobe->surface = surface;
    lobe->wi = intersect->ray.dir;
    lobe->normal = surface->normal;
    lobe->transform = Transform3f::basis_transform(lobe->normal);
    lobe->color = VEC3F_ONES;
    lobe->g = anistropy_texture->evaluate<float>(surface->u, surface->v, surface->w, intersect, resources);
    lobe->g_sqr = lobe->g * lobe->g;
    lobe->g_inv = 1.f / lobe->g;
    instance.push(lobe);
    return instance;
}

bool MaterialLobeHenyeyGreenstein::sample(MaterialSample& sample, Resources& resources) const
{
    const float * rnd = rng.rand();

    const float theta = TWO_PI * rnd[0];
    float cos_phi = 0.f;
    float weight = 0.f;
    if(g > EPSILON_F || g < -EPSILON_F)
    {
        float a = (1.f - g_sqr) / (1.f - g + 2.f * g * rnd[1]);
        cos_phi = (0.5f * g_inv) * (1.f + g_sqr - a * a);
        weight = INV_FOUR_PI * (1.f - g_sqr) / std::pow(1.f + g_sqr - 2.f * g * cos_phi, 1.5f);
    }
    else
    {
        cos_phi = 1.f - 2.f * rnd[1];
        weight = INV_FOUR_PI;
    }
    float sin_phi = sqrtf(std::max(EPSILON_F, 1.f - cos_phi * cos_phi));
    sample.wo = transform * Vec3f(sin_phi * cosf(theta), cos_phi, sin_phi * sinf(theta));
    sample.value = weight; // * color;
    sample.pdf = weight;
    return true;
}

bool MaterialLobeHenyeyGreenstein::evaluate(MaterialSample& sample, Resources& resources) const
{
    float cos_phi = sample.wo.dot(normal);
    float weight = 0.f;
    if(g > EPSILON_F || g < -EPSILON_F)
        weight = INV_FOUR_PI * (1.f - g_sqr) / std::pow(1.f + g_sqr - 2.f * g * cos_phi, 1.5f);
    else
        weight = INV_FOUR_PI;
    sample.value = weight; // * color;
    sample.pdf = weight;
    return true;
}