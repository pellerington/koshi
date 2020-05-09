#include <lights/LightSamplerArea.h>

LightSamplerArea::LightSamplerArea(GeometryArea * geometry) 
: geometry(geometry)
{
    light = geometry->get_attribute<Light>("light");
    double_sided = false;
    const Transform3f& obj_to_world = geometry->get_obj_to_world();
    area  = obj_to_world.multiply(Vec3f(1.f, 0.f, 0.f), false).length();
    area *= obj_to_world.multiply(Vec3f(0.f, 1.f, 0.f), false).length();
}

bool LightSamplerArea::sample_light(const uint num_samples, const Intersect * intersect, std::vector<LightSample> &light_samples, Resources &resources)
{
    //CHECK IF WE ARE ABOVE THE LIGHT AND !DOUBLE SIDED THEN RETURN FALSE

    const Transform3f& obj_to_world = geometry->get_obj_to_world();
    const Vec3f& world_normal = geometry->get_world_normal();
    RandomNumberGen2D rng = resources.random_number_service.get_random_2D();
    for(uint i = 0; i < num_samples; i++)
    {
        const Vec2f rnd = rng.rand();

        const Vec3f light_pos = obj_to_world * Vec3f(rnd[0]-0.5f, rnd[1]-0.5f, 0.f);
        const Vec3f dir = intersect->surface.position - light_pos;
        const float sqr_len = dir.sqr_length();
        const float cos_theta = world_normal.dot(dir / sqrtf(sqr_len));
        if(cos_theta < 0.f && !double_sided)
            continue;

        light_samples.emplace_back();
        LightSample &light_sample = light_samples.back();

        light_sample.position = light_pos;

        Intersect light_intersect(Ray(intersect->surface.position, -dir));
        light_intersect.geometry = geometry;
        light_intersect.surface.set(light_sample.position, world_normal, rnd[0], rnd[1], -dir);

        light_sample.intensity = light->get_intensity(&light_intersect, resources);

        light_sample.pdf = sqr_len / (area * (fabs(cos_theta) + EPSILON_F));
    }

    return true;
}

float LightSamplerArea::evaluate_light(const Intersect * light_intersect, const Intersect * intersect, Resources &resources)
{
    if(!light_intersect->surface.facing && !double_sided)
        return 0.f;

    const Vec3f& world_normal = geometry->get_world_normal();
    const Vec3f dir = intersect->surface.position - light_intersect->surface.position;
    const float sqr_len = dir.sqr_length();
    const float cos_theta = fabs(world_normal.dot(dir / sqrtf(sqr_len)));

    return sqr_len / (area * cos_theta);
}
