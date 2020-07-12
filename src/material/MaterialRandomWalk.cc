#include <material/MaterialRandomWalk.h>
#include <intersection/Intersect.h>
#include <geometry/Geometry.h>
#include <material/MaterialLambert.h>
#include <texture/TextureConstant.h>
#include <intersection/Intersector.h>

MaterialRandomWalk::MaterialRandomWalk(const Texture * color_texture, const Texture * density_texture, const float& anistropy, 
                                       const Texture * normal_texture, const Texture * opacity_texture)
: Material(normal_texture, opacity_texture), color_texture(color_texture), density_texture(density_texture), anistropy(anistropy)
{
    // TODO: Have exit color be texturable here ( so we don't have to clean it up. )
    // TODO: Plug exit_material into MaterialRandomWalk and setup so it gets deleted automatically.
    exit_material = new MaterialLambert<false>(new TextureConstant(VEC3F_ONES), normal_texture, new TextureConstant(VEC3F_ONES));
}

MaterialLobes MaterialRandomWalk::instance(const Surface * surface, const Intersect * intersect, Resources& resources)
{
    MaterialLobes instance(resources.memory);
    
    MaterialLobeRandomWalk * lobe = resources.memory->create<MaterialLobeRandomWalk>();

    lobe->rng = resources.random_service->get_random<2>();

    lobe->surface = surface;
    lobe->wi = intersect->ray.dir;
    lobe->normal = (normal_texture) ? (intersect->geometry->get_obj_to_world() * normal_texture->evaluate<Vec3f>(surface->u, surface->v, surface->w, intersect, resources)).normalized() : surface->normal;
    lobe->transform = Transform3f::basis_transform(lobe->normal);

    lobe->color = color_texture->evaluate<Vec3f>(surface->u, surface->v, surface->w, intersect, resources);
    lobe->density = density_texture->evaluate<Vec3f>(surface->u, surface->v, surface->w, intersect, resources);
    lobe->density.max(EPSILON_F);
    lobe->scatter = lobe->density * lobe->color;

    lobe->g = anistropy;
    lobe->g_sqr = lobe->g * lobe->g;
    lobe->g_inv = 1.f / lobe->g;

    lobe->intersector = resources.intersector->get_intersector(intersect->geometry);
    lobe->exit_material = exit_material;

    instance.push(lobe);
    return instance;
}