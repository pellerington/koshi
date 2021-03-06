#include <koshi/material/MaterialLambert.h>

#include <koshi/math/Helpers.h>
#include <koshi/math/Color.h>
#include <cmath>
#include <iostream>

#include <koshi/geometry/Geometry.h>

template<bool REFLECT>
MaterialLambert<REFLECT>::MaterialLambert(const Texture * color_texture, const Texture * normal_texture, const Texture * opacity_texture)
: Material(normal_texture, opacity_texture), color_texture(color_texture)
{
}

template<bool REFLECT>
MaterialLobes MaterialLambert<REFLECT>::instance(const Surface * surface, const Intersect * intersect, Resources& resources)
{
    MaterialLobes instance(resources.memory);
    MaterialLobeLambert<REFLECT> * lobe = resources.memory->create<MaterialLobeLambert<REFLECT>>();
    
    lobe->rng = resources.random_service->get_random<2>();

    lobe->surface = surface;
    lobe->wi = intersect->ray.dir;
    lobe->normal = (normal_texture) ? (intersect->geometry->get_obj_to_world() * normal_texture->evaluate<Vec3f>(surface->u, surface->v, surface->w, intersect, resources)).normalized() : surface->normal;
    lobe->transform = Transform3f::basis_transform(lobe->normal);

    lobe->color = color_texture->evaluate<Vec3f>(surface->u, surface->v, surface->w, intersect, resources);
    instance.push(lobe);
    return instance;
}

template<bool REFLECT>
bool MaterialLobeLambert<REFLECT>::sample(MaterialSample& sample, Resources& resources) const
{
    if(REFLECT && !surface->facing) return false;

    const float * rnd = rng.rand();
    const float theta = TWO_PI * rnd[0];
#if UNIFORM_SAMPLE
    const float phi = acosf(rnd[1]);
    sample.wo = transform * Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta));
    sample.pdf = INV_TWO_PI;
#else
    const float r = sqrtf(rnd[1]);
    const float x = r * cosf(theta), z = r * sinf(theta), y = sqrtf(std::max(EPSILON_F, 1.f - rnd[1]));
    sample.wo = transform * Vec3f(x, y, z);
    sample.pdf = y * INV_PI;
#endif
    sample.value = color * INV_PI * sample.wo.dot(normal);
    if(!REFLECT && surface->facing) sample.wo = -sample.wo; 

    return true;
}

template<bool REFLECT>
bool MaterialLobeLambert<REFLECT>::evaluate(MaterialSample& sample, Resources& resources) const
{
    if(REFLECT && !surface->facing)
        return false;

    const float n_dot_wo = normal.dot(sample.wo);
    const float n_dot_wi = normal.dot(wi);

    if((REFLECT && n_dot_wo * n_dot_wi > 0.f) || (!REFLECT && n_dot_wo * n_dot_wi < 0.f))
        return false;

    sample.value = color * INV_PI * fabs(n_dot_wo);

    #if UNIFORM_SAMPLE
        sample.pdf = INV_TWO_PI;
    #else
        sample.pdf = fabs(n_dot_wo) * INV_PI;
    #endif

    return true;
}