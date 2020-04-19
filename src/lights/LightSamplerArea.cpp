#include <lights/LightSamplerArea.h>

LightSamplerArea::LightSamplerArea(GeometryArea * geometry) 
: geometry(geometry)
{
    light = geometry->light.get();
    double_sided = false;
    const Transform3f& obj_to_world = geometry->get_obj_to_world();
    area  = obj_to_world.multiply(Vec3f(1.f, 0.f, 0.f), false).length();
    area *= obj_to_world.multiply(Vec3f(0.f, 1.f, 0.f), false).length();
}

bool LightSamplerArea::sample_light(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources)
{
    //CHECK IF WE ARE ABOVE THE LIGHT AND !DOUBLE SIDED THEN RETURN FALSE

    const Transform3f& obj_to_world = geometry->get_obj_to_world();
    const Vec3f& world_normal = geometry->get_world_normal();
    RNG &rng = resources.rng; rng.Reset2D();
    for(uint i = 0; i < num_samples; i++)
    {
        const Vec2f rnd = rng.Rand2D();

        const Vec3f light_pos = obj_to_world * Vec3f(rnd[0]*2.f-1.f, rnd[1]*2.f-1.f, 0.f);
        const Vec3f dir = *pos - light_pos;
        const float sqr_len = dir.sqr_length();
        const float cos_theta = world_normal.dot(dir / sqrtf(sqr_len));
        if(cos_theta < 0.f && !double_sided)
            continue;

        light_samples.emplace_back();
        LightSample &light_sample = light_samples.back();

        light_sample.position = light_pos;
        light_sample.intensity = light->get_emission();
        light_sample.pdf = sqr_len / (area * (fabs(cos_theta) + EPSILON_F));
    }

    return true;
}

bool LightSamplerArea::evaluate_light(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources)
{
    if(!intersect.front && !double_sided)
        return false;

    const Vec3f& world_normal = geometry->get_world_normal();

    light_sample.position = intersect.position;
    light_sample.intensity = light->get_emission();

    const Vec3f dir = *pos - light_sample.position;
    const float sqr_len = dir.sqr_length();
    const float cos_theta = fabs(world_normal.dot(dir / sqrtf(sqr_len)));

    light_sample.pdf = sqr_len / (area * cos_theta);

    return true;
}
