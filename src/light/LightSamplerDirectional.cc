#include <light/LightSamplerDirectional.h>
#include <geometry/SurfaceDistant.h>
#include <material/Material.h>

LightSamplerDirectional::LightSamplerDirectional(Geometry * geometry)
: geometry(geometry)
{
    direction = geometry->get_obj_to_world().multiply(Vec3f(0.f,1.f,0.f), false).normalized();
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

    sample.position = surface->position + direction*INV_EPSILON_F;

    Intersect light_intersect(Ray(surface->position, direction));
    light_intersect.geometry = geometry;
    // TODO: Add uv once we have angle availible
    SurfaceDistant * distant = resources.memory->create<SurfaceDistant>(0.f, 0.f, 0.f);
    light_intersect.geometry_data = distant;

    sample.intensity = INV_EPSILON_F * material->emission(distant->u, distant->v, distant->w, &light_intersect, resources);
    sample.pdf = INV_EPSILON_F;

    return true;
}