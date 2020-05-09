#include <materials/MaterialDielectric.h>

#include <materials/MaterialGGXReflect.h>
#include <materials/MaterialGGXRefract.h>

#include <Math/Helpers.h>
#include <Util/Color.h>
#include <cmath>
#include <iostream>

MaterialDielectric::MaterialDielectric(const AttributeVec3f &reflective_color_attribute,
                                       const AttributeVec3f &refractive_color_attribute,
                                       const AttributeFloat &roughness_attribute,
                                       const float &ior)
: reflective_color_attribute(reflective_color_attribute), 
  refractive_color_attribute(refractive_color_attribute),
  roughness_attribute(roughness_attribute), ior(ior)
{
}

MaterialInstance MaterialDielectric::instance(const GeometrySurface * surface, Resources &resources)
{
    MaterialInstance instance;

    MaterialLobeGGXRefract * refract_lobe = resources.memory.create<MaterialLobeGGXRefract>();
    refract_lobe->surface = surface;
    refract_lobe->rng = resources.random_number_service.get_random_2D();
    refract_lobe->ior_in = 1.f;  //surface->facing ? surface->curr_ior : ior;
    refract_lobe->ior_out = ior; //surface->facing ? ior : surface->prev_ior;
    refract_lobe->color = refractive_color_attribute.get_value(surface->u, surface->v, 0.f, resources);
    refract_lobe->roughness = roughness_attribute.get_value(surface->u, surface->v, 0.f, resources);
    refract_lobe->roughness = clamp(refract_lobe->roughness * refract_lobe->roughness, 0.01f, 0.99f);
    refract_lobe->roughness_sqr = refract_lobe->roughness * refract_lobe->roughness;
    refract_lobe->fresnel = resources.memory.create<FresnelDielectric>(refract_lobe->ior_in, refract_lobe->ior_out);

    MaterialLobeGGXReflect * reflect_lobe = resources.memory.create<MaterialLobeGGXReflect>();
    reflect_lobe->surface = surface;
    reflect_lobe->rng = resources.random_number_service.get_random_2D();
    reflect_lobe->color = reflective_color_attribute.get_value(surface->u, surface->v, 0.f, resources);
    reflect_lobe->roughness = refract_lobe->roughness;
    reflect_lobe->roughness_sqr = refract_lobe->roughness_sqr;
    reflect_lobe->fresnel = refract_lobe->fresnel;

    instance.push(refract_lobe);
    instance.push(reflect_lobe);

    return instance;
}