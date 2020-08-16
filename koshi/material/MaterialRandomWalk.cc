#include <koshi/material/MaterialRandomWalk.h>
#include <koshi/intersection/Intersect.h>
#include <koshi/geometry/Geometry.h>
#include <koshi/material/MaterialLambert.h>
#include <koshi/texture/TextureConstant.h>
#include <koshi/intersection/Intersector.h>
#include <koshi/integrator/Integrator.h>

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
    lobe->max_density = lobe->density.max();
    lobe->inv_max_density = 1.f / lobe->max_density;
    lobe->null = lobe->max_density - lobe->density;
    lobe->scatter = lobe->density * lobe->color;
    lobe->g = anistropy;
    lobe->g_sqr = lobe->g * lobe->g;
    lobe->g_inv = 1.f / lobe->g;

    lobe->intersector = resources.intersector->get_intersector(intersect->geometry);
    lobe->exit_material = exit_material;
    lobe->next_path.depth = intersect->path->depth + 1;
    lobe->next_path.quality = intersect->path->quality / 128;
    lobe->next_path.prev_path = intersect->path;

    instance.push(lobe);
    return instance;
}

bool MaterialLobeRandomWalk::sample(MaterialSample& sample, Resources& resources) const
{
    // Sample an initial direction.
    const float * rnd = rng.rand();
    const float theta = TWO_PI * rnd[0];
    const float r = sqrtf(rnd[1]);
    const float x = r * cosf(theta), z = r * sinf(theta), y = sqrtf(std::max(EPSILON_F, 1.f - rnd[1]));
    Vec3f wo = transform * -Vec3f(x, y, z);                

    // Setup wieghts and pdf.
    Vec3f weight = VEC3F_ONES;
    Vec3f position = surface->back_position;

    // Find scatter postion.
    uint bounces = 0;
    while(bounces++ < 32)
    {
        // Sample a length
        const float tmax = -logf(resources.random_service->rand()) * inv_max_density;

        // Intersect the sampled tmax.
        Ray ray(position, wo, 0.f, tmax);
        IntersectList * intersects = intersector->intersect(ray, &next_path, resources, &intersection_callback);
        if(intersects->hit())
        {
            ((Surface *)intersects->get(0)->geometry_data)->material = exit_material;
            sample.value = weight * Integrator::shade(intersects, resources);
            sample.pdf = 1.f;
            return true;
        }

        // Update the position;
        position = ray.get_position(tmax);

        // Event probabilities.
        float nprob = (weight * null).avg();
        float sprob = (weight * scatter).avg();
        const float inv_sum = 1.f / (nprob + sprob);
        nprob *= inv_sum; sprob *= inv_sum;

        // Null event
        if(resources.random_service->rand() < nprob)
        {
            weight *= null * inv_max_density / nprob;
        }
        // Scatter event
        else
        {
            const float theta = TWO_PI * resources.random_service->rand();
            float cos_phi = 0.f;
            if(g > EPSILON_F || g < -EPSILON_F)
            {
                float a = (1.f - g_sqr) / (1.f - g + 2.f * g * resources.random_service->rand());
                cos_phi = (0.5f * g_inv) * (1.f + g_sqr - a * a);
            }
            else
            {
                cos_phi = 1.f - 2.f * resources.random_service->rand();
            }
            float sin_phi = sqrtf(std::max(EPSILON_F, 1.f - cos_phi * cos_phi));
            const float x = sin_phi * cosf(theta), z = sin_phi * sinf(theta), y = cos_phi;
            wo = Transform3f::basis_transform(wo) * Vec3f(x, y, z);

            weight *= scatter * inv_max_density / sprob;
        }
    }

    return false;

}

void MaterialLobeRandomWalk::post_intersection_callback(IntersectList * intersects, void * data, Resources& resources)
{
    for(uint i = 0; i < intersects->size(); i++)
        if(((Surface*)intersects->get(i)->geometry_data)->facing)
            i = intersects->pop(i) - 1;

    if(intersects->size() > 1)
    {
        float tmin = FLT_MAX;
        for(uint i = 0; i < intersects->size(); i++)
            tmin = std::min(tmin, intersects->get(i)->t);
        for(uint i = 0; i < intersects->size(); i++)
            if(intersects->get(i)->t > tmin)
                i = intersects->pop(i) - 1;
    }
}

IntersectionCallbacks MaterialLobeRandomWalk::intersection_callback(nullptr, nullptr, MaterialLobeRandomWalk::post_intersection_callback, nullptr);
