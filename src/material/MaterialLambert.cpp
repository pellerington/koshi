#include <material/MaterialLambert.h>

#include <Math/Helpers.h>
#include <Util/Color.h>
#include <cmath>
#include <iostream>

#include <geometry/Geometry.h>

template<bool REFLECT>
MaterialLambert<REFLECT>::MaterialLambert(const AttributeVec3f& color_attr)
: color_attr(color_attr)
{
}

template<bool REFLECT>
MaterialInstance MaterialLambert<REFLECT>::instance(const Surface * surface, const Intersect * intersect, Resources &resources)
{
    MaterialInstance instance(resources.memory);
    MaterialLobeLambert<REFLECT> * lobe = resources.memory->create<MaterialLobeLambert<REFLECT>>();
    
    lobe->rng = resources.random_service->get_random_2D();

    lobe->surface = surface;
    lobe->wi = intersect->ray.dir;
    if(!intersect->geometry->eval_geometry_attribute(lobe->normal, "normals", surface->u, surface->v, surface->w, intersect->geometry_primitive, resources))
        lobe->normal = surface->normal;
    else
        lobe->normal = (intersect->geometry->get_obj_to_world() * lobe->normal).normalized();
    lobe->transform = Transform3f::basis_transform(lobe->normal);

    lobe->color = color_attr.get_value(surface->u, surface->v, surface->w, resources);
    instance.push(lobe);
    return instance;
}

template<bool REFLECT>
bool MaterialLobeLambert<REFLECT>::sample(MaterialSample& sample, Resources &resources) const
{
    if(REFLECT && !surface->facing)
        return false;

    const Vec2f rnd = rng.rand();
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

    sample.weight = color * INV_PI * sample.wo.dot(normal);
    if(!REFLECT && surface->facing)
        sample.wo = -sample.wo; 

    return true;
}

template<bool REFLECT>
Vec3f MaterialLobeLambert<REFLECT>::weight(const Vec3f& wo, Resources &resources) const
{
    // TODO: Add domain check to helpers.
    if(REFLECT && !surface->facing)
        return VEC3F_ZERO;

    const float n_dot_wo = normal.dot(wo);
    const float n_dot_wi = normal.dot(wi);
    if((REFLECT && n_dot_wo < 0.f) || (!REFLECT && n_dot_wo * n_dot_wi > 0.f))
        return VEC3F_ZERO;

    return color * INV_PI * fabs(n_dot_wo);
}

template<bool REFLECT>
float MaterialLobeLambert<REFLECT>::pdf(const Vec3f& wo, Resources &resources) const
{
    // TODO: Add domain check to helpers.
    if(REFLECT && !surface->facing)
        return 0.f;

    const float n_dot_wo = normal.dot(wo);
    const float n_dot_wi = normal.dot(wi);
    if((REFLECT && n_dot_wo < 0.f) || (!REFLECT && n_dot_wo * n_dot_wi > 0.f))
        return 0.f;

#if UNIFORM_SAMPLE
    return INV_TWO_PI;
#else
    return fabs(n_dot_wo) * INV_PI;
#endif
}