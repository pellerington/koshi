#include <material/MaterialDielectric.h>

#include <material/MaterialGGXReflect.h>
#include <material/MaterialGGXRefract.h>
#include <integrators/AbsorbtionMedium.h>

#include <Math/Helpers.h>
#include <Util/Color.h>
#include <cmath>
#include <iostream>

MaterialDielectric::MaterialDielectric(const AttributeVec3f &reflective_color_attribute,
                                       const AttributeVec3f &refractive_color_attribute,
                                       const float& refractive_color_depth,
                                       const AttributeFloat &roughness_attribute,
                                       const float &ior)
: reflective_color_attribute(reflective_color_attribute), 
  refractive_color_attribute(refractive_color_attribute),
  refractive_color_depth(refractive_color_depth),
  roughness_attribute(roughness_attribute), ior(ior)
{
}

MaterialInstance MaterialDielectric::instance(const Surface * surface, const Intersect * intersect, Resources &resources)
{
    MaterialInstance instance(resources.memory);

    MaterialLobeGGXRefract * refract_lobe = resources.memory->create<MaterialLobeGGXRefract>();
    refract_lobe->rng = resources.random_service->get_random_2D();
    
    refract_lobe->surface = surface;
    refract_lobe->wi = intersect->ray.dir;
    refract_lobe->normal = ((surface->facing) ? surface->normal : -surface->normal);
    refract_lobe->transform = Transform3f::basis_transform(refract_lobe->normal);

    refract_lobe->ior_in = 1.f;  //surface->facing ? surface->curr_ior : ior;
    refract_lobe->ior_out = ior; //surface->facing ? ior : surface->prev_ior;
    refract_lobe->color = refractive_color_attribute.get_value(surface->u, surface->v, surface->w, resources);
    refract_lobe->roughness = roughness_attribute.get_value(surface->u, surface->v, surface->w, resources);
    refract_lobe->roughness = clamp(refract_lobe->roughness * refract_lobe->roughness, 0.01f, 0.99f);
    refract_lobe->roughness_sqr = refract_lobe->roughness * refract_lobe->roughness;
    refract_lobe->fresnel = resources.memory->create<FresnelDielectric>(refract_lobe->ior_in, refract_lobe->ior_out);
    if(refractive_color_depth > EPSILON_F)
    {
        refract_lobe->interior = surface->facing ? resources.memory->create<AbsorbtionMedium>(refract_lobe->color, refractive_color_depth) : nullptr;
        refract_lobe->color = VEC3F_ONES;
    }

    MaterialLobeGGXReflect * reflect_lobe = resources.memory->create<MaterialLobeGGXReflect>();
    reflect_lobe->rng = resources.random_service->get_random_2D();
   
    reflect_lobe->surface = surface;
    reflect_lobe->wi = intersect->ray.dir;
    reflect_lobe->normal = refract_lobe->normal;
    reflect_lobe->transform = refract_lobe->transform;

    reflect_lobe->color = reflective_color_attribute.get_value(surface->u, surface->v, surface->w, resources);
    reflect_lobe->roughness = refract_lobe->roughness;
    reflect_lobe->roughness_sqr = refract_lobe->roughness_sqr;
    reflect_lobe->fresnel = refract_lobe->fresnel;

    instance.push(refract_lobe);
    instance.push(reflect_lobe);

    return instance;
}
