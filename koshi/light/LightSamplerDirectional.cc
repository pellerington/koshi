#include <koshi/light/LightSamplerDirectional.h>
#include <koshi/geometry/SurfaceDistant.h>
#include <koshi/material/Material.h>

LightSamplerDirectional::LightSamplerDirectional(GeometryDirectional * geometry)
: geometry(geometry)
{
}

void LightSamplerDirectional::pre_render(Resources& resources)
{
    material = geometry->get_attribute<Material>("material");
}

const LightSamplerData * LightSamplerDirectional::pre_integrate(const Surface * surface, Resources& resources) const
{
    LightSamplerDataDirectional * data = resources.memory->create<LightSamplerDataDirectional>();
    data->surface = surface;
    data->rng = resources.random_service->get_random<2>();
    return data;
}

bool LightSamplerDirectional::sample(LightSample& sample, const LightSamplerData * data, Resources& resources) const
{
    const LightSamplerDataDirectional * light_data = (const LightSamplerDataDirectional *)data;
    const Surface * surface = light_data->surface;

    const float * rnd = light_data->rng.rand();
    const float theta = TWO_PI * rnd[0];
    const float phi = acosf(rnd[1] - rnd[1] * geometry->get_cos_phi_max() + geometry->get_cos_phi_max());

    Vec3f direction = geometry->get_obj_to_world().multiply(Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta)), false);

    sample.position = surface->position + direction*INV_EPSILON_F;

    Intersect light_intersect(Ray(surface->position, direction));
    light_intersect.geometry = geometry;
    SurfaceDistant * distant = resources.memory->create<SurfaceDistant>(theta * INV_TWO_PI, phi / geometry->get_phi_max(), 0.f);
    light_intersect.geometry_data = distant;

    sample.intensity = material->emission(distant->u, distant->v, distant->w, &light_intersect, resources);
    sample.pdf = 1.f / geometry->get_area();

    return true;
}

float LightSamplerDirectional::evaluate(const Intersect * intersect, const LightSamplerData * data, Resources& resources) const
{
    return 1.f / geometry->get_area();
}