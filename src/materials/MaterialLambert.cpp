#include <materials/MaterialLambert.h>

#include <Math/Helpers.h>
#include <Util/Color.h>
#include <cmath>
#include <iostream>

template<bool FRONT>
MaterialLambert<FRONT>::MaterialLambert(const AttributeVec3f &diffuse_color_attr)
: diffuse_color_attr(diffuse_color_attr)
{
}

template<bool FRONT>
MaterialInstance MaterialLambert<FRONT>::instance(const GeometrySurface * surface, Resources &resources)
{
    MaterialInstance instance;
    MaterialLobeLambert<FRONT> * lobe = resources.memory.create<MaterialLobeLambert<FRONT>>();
    lobe->surface = surface;
    lobe->rng = resources.random_number_service.get_random_2D();
    lobe->color = diffuse_color_attr.get_value(surface->u, surface->v, 0.f, resources);
    instance.push(lobe);
    return instance;
}

template<bool FRONT>
bool MaterialLobeLambert<FRONT>::sample(MaterialSample& sample, Resources &resources) const
{
    if(FRONT && !surface->facing)
        return false;

    const Vec2f rnd = rng.rand();
    const float theta = TWO_PI * rnd[0];

#if UNIFORM_SAMPLE
    const float phi = acosf(rnd[1]);
    sample.wo = surface->transform * Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta));
    sample.pdf = INV_TWO_PI;
#else
    const float r = sqrtf(rnd[1]);
    const float x = r * cosf(theta), z = r * sinf(theta), y = sqrtf(std::max(EPSILON_F, 1.f - rnd[1]));
    sample.wo = surface->transform * Vec3f(x, y, z);
    sample.pdf = y * INV_PI;
#endif
    sample.weight = color * INV_PI * sample.wo.dot(surface->normal);
    if(!FRONT && surface->facing)
        sample.wo = -sample.wo; 

    return true;
}

template<bool FRONT>
bool MaterialLobeLambert<FRONT>::evaluate(MaterialSample& sample, Resources &resources) const
{
    if(FRONT && !surface->facing)
        return false;

    float n_dot_wo = surface->normal.dot(sample.wo);
    if((FRONT && n_dot_wo < 0.f) || (!FRONT && n_dot_wo * surface->n_dot_wi > 0.f))
        return false;

    n_dot_wo = fabs(n_dot_wo);
    sample.weight = color * INV_PI * n_dot_wo;
#if UNIFORM_SAMPLE
    sample.pdf = INV_TWO_PI;
#else
    sample.pdf = n_dot_wo * INV_PI;
#endif

    return true;
}
