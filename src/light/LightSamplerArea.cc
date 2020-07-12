#include <light/LightSamplerArea.h>
#include <material/Material.h>

LightSamplerArea::LightSamplerArea(GeometryArea * geometry) 
: geometry(geometry)
{
    double_sided = false;
    const Transform3f& obj_to_world = geometry->get_obj_to_world();
    area  = obj_to_world.multiply(Vec3f(1.f, 0.f, 0.f), false).length();
    area *= obj_to_world.multiply(Vec3f(0.f, 1.f, 0.f), false).length();

    normal = obj_to_world.multiply(Vec3f(0.f, 0.f, -1.f), false).normalized();
}

void LightSamplerArea::pre_render(Resources& resources)
{
    material = geometry->get_attribute<Material>("material");
}

const LightSamplerData * LightSamplerArea::pre_integrate(const Surface * surface, Resources& resources) const
{
    LightSamplerDataArea * data = resources.memory->create<LightSamplerDataArea>();
    data->surface = surface;
    data->rng = resources.random_service->get_random<2>();
    return data;
}

bool LightSamplerArea::sample(LightSample& sample, const LightSamplerData * data, Resources& resources) const
{
    const LightSamplerDataArea * light_data = (const LightSamplerDataArea*)data;
    const Surface * surface = light_data->surface;
    const Transform3f& obj_to_world = geometry->get_obj_to_world();

    const float * rnd = light_data->rng.rand();
    sample.position = obj_to_world * Vec3f(AREA_LENGTH*(rnd[0]-0.5f), AREA_LENGTH*(rnd[1]-0.5f), 0.f);
    const Vec3f dir = sample.position - surface->position;
    const float sqr_len = dir.sqr_length();
    const float cos_theta = normal.dot(dir / sqrtf(sqr_len));
    if(cos_theta > 0.f && !double_sided)
        return false;

    Intersect light_intersect(Ray(surface->position, dir));
    light_intersect.geometry = geometry;
    Surface * light_surface = resources.memory->create<Surface>(sample.position, normal, rnd[0], rnd[1], 0.f, cos_theta < 0.f);
    light_intersect.geometry_data = light_surface;
    sample.intensity = material->emission(light_surface->u, light_surface->v, light_surface->w, &light_intersect, resources);
    sample.pdf = sqr_len / (area * (fabs(cos_theta) + EPSILON_F));

    return true;
}

float LightSamplerArea::evaluate(const Intersect * intersect, const LightSamplerData * data, Resources& resources) const
{
    const LightSamplerDataArea * light_data = (const LightSamplerDataArea*)data;
    const Surface * surface = light_data->surface;
    const Surface * light_surface = (const Surface*)intersect->geometry_data;
    if(!light_surface->facing && !double_sided)
        return 0.f;

    const Vec3f dir = light_surface->position - surface->position;
    const float sqr_len = dir.sqr_length();
    const float cos_theta = normal.dot(dir / sqrtf(sqr_len));

    return sqr_len / (area * (fabs(cos_theta) + EPSILON_F));
}
